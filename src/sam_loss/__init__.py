import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
import torch
import torch.nn.functional as F
from torch import nn

import sam_loss
from sam_loss.sam2.sam2_image_predictor import SAM2ImagePredictor

SAMDistanceType = Literal['l1', 'l2', 'cos']


@dataclass
class SAMLossConfig:
    sam_distance: SAMDistanceType = 'l1'
    sam_model: str = 'facebook/sam2.1-hiera-small'


class SAMLoss(nn.Module):

    def __init__(self, config: SAMLossConfig = SAMLossConfig()):
        super().__init__()
        self._config = config

        sys.path.append(str(Path(sam_loss.__file__).parent))
        hydra.initialize()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        # reinit hydra with a new search path for configs
        hydra.initialize_config_module("sam_loss.sam2", version_base='1.2')
        sam_model = SAM2ImagePredictor.from_pretrained(config.sam_model)
        sam_model.model.eval()
        for p in sam_model.model.parameters():
            p.requires_grad = False
        self._sam_model = sam_model

    def forward(self, predicted_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            predicted_images: [B, 3, H, W] in [0, 1] range
            target_images: [B, 3, H, W] in [0, 1] range

        Returns
        -------
            scalar with SAM loss
        """

        self._sam_model.set_image_batch_backprop(predicted_images)  # TODO: configrm (BCHW) in (0,1) range
        rendered_features_list = [self._sam_model.get_all_image_features()['image_embed']]

        with torch.no_grad():
            self._sam_model.set_image_batch_backprop(target_images.to(dtype=predicted_images.dtype))  # TODO: configrm (BCHW) in (0,1) range
            target_features_list = [self._sam_model.get_all_image_features()['image_embed']]

        sam_loss_combined = torch.tensor(0.0, dtype=torch.float32, device=predicted_images.device)
        for i in range(len(rendered_features_list)):
            rendered_features, target_features = rendered_features_list[i], target_features_list[i]
            if self._config.sam_distance == "l1":
                sam_losses = F.l1_loss(
                    rendered_features,
                    target_features.detach(),
                    reduction="mean"
                )
            elif self._config.sam_distance == "l2":
                sam_losses = F.mse_loss(
                    rendered_features,
                    target_features.detach(),
                    reduction="mean"
                )
            elif self._config.sam_distance == 'cos':
                rendered_features = F.normalize(rendered_features, dim=1)
                target_features = F.normalize(target_features.detach(), dim=1)
                sam_losses = 1 - (rendered_features * target_features).sum(dim=1).mean()
            else:
                raise ValueError(f"Unknown sam loss distance: {self._config.sam_distance}")
            sam_loss_combined += sam_losses

        return sam_loss_combined
