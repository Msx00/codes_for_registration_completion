import sys
import unittest
from pathlib import Path

import torch


PIVOTS_ROOT = Path(__file__).resolve().parents[1]
if str(PIVOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PIVOTS_ROOT))

from models.P_V2S_Net_Full2Full_V1 import PV2SNetFull2FullV1
from models.global_matcher import GlobalMatcher


class Full2FullV1InitializationTest(unittest.TestCase):
    def test_nearby_cloud_initialization_stays_near_source(self):
        torch.manual_seed(7)
        model = PV2SNetFull2FullV1(
            global_spatial_sigma=0.2,
            max_coarse_flow_normalized=0.25,
            num_refinement_steps=3,
        ).eval()
        source = torch.rand(2, 256, 3) * 2.0 - 1.0
        target = source + 0.01 * torch.randn_like(source)

        with torch.no_grad():
            output = model(source, target)

        coarse_prediction = output["warped_source_stages"][0]
        source_rmse = torch.sqrt(
            (source - target).square().sum(dim=-1).mean()
        )
        coarse_rmse = torch.sqrt(
            (coarse_prediction - target).square().sum(dim=-1).mean()
        )
        coarse_flow_magnitude = output["flow_stages"][0].norm(
            dim=-1
        ).mean()
        confidence = output["global_match_confidence"].mean()

        self.assertTrue(
            all(torch.isfinite(stage).all() for stage in output["flow_stages"])
        )
        self.assertLessEqual(coarse_rmse, 1.10 * source_rmse + 1e-6)
        self.assertLess(coarse_flow_magnitude, 0.01)
        self.assertTrue(torch.isfinite(confidence))
        # Zero-initialized recurrent residuals must not alter the gated coarse flow.
        for stage in output["flow_stages"][1:]:
            self.assertTrue(torch.equal(stage, output["flow_stages"][0]))

    def test_uniform_assignment_falls_back_to_zero_flow(self):
        matcher = GlobalMatcher(
            feature_dim=8,
            projection_dim=4,
            spatial_sigma=0.2,
            max_coarse_flow=0.25,
        ).eval()
        source = torch.zeros(1, 8, 3)
        target = torch.ones(1, 8, 3)
        source_features = torch.zeros(1, 8, 8)
        target_features = torch.zeros(1, 8, 8)

        with torch.no_grad():
            output = matcher(
                source, target, source_features, target_features
            )

        self.assertTrue(torch.isfinite(output["assignment"]).all())
        self.assertTrue(torch.equal(output["coarse_flow"], torch.zeros_like(source)))

    def test_zero_initialized_geometry_and_residual_heads(self):
        model = PV2SNetFull2FullV1().eval()
        self.assertTrue(
            torch.equal(
                model.global_matcher.geometry_mlp[-1].weight,
                torch.zeros_like(model.global_matcher.geometry_mlp[-1].weight),
            )
        )
        self.assertTrue(
            torch.equal(
                model.iterative_refiner.local_refiner.residual_head[-1].weight,
                torch.zeros_like(
                    model.iterative_refiner.local_refiner.residual_head[-1].weight
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
