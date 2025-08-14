import sys
from pathlib import Path
from unittest import TestCase

import hydra
import torch

import sam_loss
from sam_loss import SAMLoss
from sam_loss.sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Test(TestCase):
    def test_sam2(self):
        sys.path.append(str(Path(sam_loss.__file__).parent))
        hydra.initialize()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        # reinit hydra with a new search path for configs
        hydra.initialize_config_module("sam_loss.sam2", version_base='1.2')
        loss_sam_model = 'facebook/sam2.1-hiera-small'
        sam_model = SAM2ImagePredictor.from_pretrained(loss_sam_model)
        sam_model.model.eval()
        print('hi')

        predicted_images = torch.randn((2, 3, 512, 512), device='cuda')
        sam_model.set_image_batch_backprop(predicted_images)  # TODO: configrm (BCHW) in (0,1) range
        rendered_features_list = [sam_model.get_all_image_features()['image_embed']]

    def test_sam2_loss(self):
        predicted_images = torch.randn((2, 3, 512, 512), device='cuda')
        target_images = torch.randn((2, 3, 512, 512), device='cuda')
        sam_criterion = SAMLoss()
        result = sam_criterion.forward(predicted_images, target_images)
        print(result)