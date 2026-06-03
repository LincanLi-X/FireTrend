import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.firetrend_model import FireTrendModel
from modules.losses import FireTrendLoss
from utils.data_loader import create_dataloader
from utils.metrics import compute_metrics


class FireTrendCoreTest(unittest.TestCase):
    def test_model_loss_and_metrics_smoke(self):
        torch.manual_seed(0)
        batch, steps, height, width = 2, 4, 6, 7
        model = FireTrendModel(
            in_dims={"fire": 1, "meteo": 8, "geo": 10},
            embed_dim=16,
            num_heads=4,
            hidden_dim=32,
            height=height,
            width=width,
            max_temporal_cells=16,
            max_spatial_anchors=8,
            max_cross_samples=64,
        )
        x_fire = torch.randn(batch, steps, 1, height, width)
        x_meteo = torch.randn(batch, steps, 8, height, width)
        x_geo = torch.randn(batch, steps, 10, height, width)
        drivers = torch.randn(batch, steps, 4, height, width)
        y_class = torch.randint(0, 3, (batch, height, width))

        outputs = model(x_fire, x_meteo, x_geo, drivers)
        self.assertEqual(tuple(outputs["logits"].shape), (batch, 3, height, width))
        self.assertGreaterEqual(float(outputs["L_contrast"].detach()), 0.0)
        self.assertGreaterEqual(float(outputs["L_pyro"].detach()), 0.0)

        losses = FireTrendLoss(class_weights=[1.0, 2.0, 3.0])(outputs, y_class, stage="joint")
        self.assertTrue(torch.isfinite(losses["L_total"]))

        metrics = compute_metrics(outputs["logits"], y_class)
        self.assertIn("IoU", metrics)
        self.assertIn("AUPRC", metrics)

    def test_firecast_loader_smoke(self):
        data_root = os.path.join(PROJECT_ROOT, "data_v2")
        loader = create_dataloader(
            data_root,
            region="california",
            seq_length=4,
            batch_size=1,
            num_workers=0,
            split="train",
            shuffle=False,
        )
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["x"].shape[1:3]), (4, 19))
        self.assertEqual(tuple(batch["drivers"].shape[1:3]), (4, 4))
        self.assertEqual(tuple(batch["y_class"].shape[1:]), (loader.dataset.height, loader.dataset.width))
        self.assertEqual(len(loader.dataset.class_weights), 3)


if __name__ == "__main__":
    unittest.main()
