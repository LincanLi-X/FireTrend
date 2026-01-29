"""
FireTrend: Main training & testing script.
Integrates wildfire (.h5) and ERA5 (.nc) datasets with FireTrend model.
Supports multi-region training (California / Florida).

Run script:
    python main.py --config config.yaml --train --region california
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Project modules ---
from modules.firetrend_model import FireTrendModel
from modules.losses import FireTrendLoss
from utils.metrics import compute_metrics
from utils.logger import get_logger
from utils.seed_utils import set_seed
from utils.data_loader import create_dataloader


# ============================================================
# Argument Parser
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(description="FireTrend Main Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints")
    parser.add_argument("--results_dir", type=str, default="./outputs/predictions")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run model evaluation / testing")
    parser.add_argument("--region", type=str, default=None, choices=["california", "florida"],
                        help="Select dataset region (overrides config.yaml)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ============================================================
# Training Function
# ============================================================
def train(model, dataloader, optimizer, loss_fn, epoch, device, logger, region):
    model.train()
    total_loss = 0.0

    for batch_idx, (X, Y_true) in enumerate(tqdm(dataloader, desc=f"[{region}] Epoch {epoch}")):
        X, Y_true = X.to(device), Y_true.to(device)

        # # Split input features [B, T, 9, H, W] = fire(1) + meteo(6) + geo(2)
        # X_fire = X[:, :, 0:1, :, :]      # wildfire history
        # X_meteo = X[:, :, 1:7, :, :]     # meteorological features
        # X_geo = X[:, :, 7:, :, :]        # dynamic geo features [B, T, 2, H, W]

        # Input shape: [B, T, 4, 9, H, W]
        X_fire  = X[:, :, 0, 0:1, :, :]     # wildfire [B, T, 4, 1, H, W]
        X_meteo = X[:, :, :, 1:7, :, :]     # meteorology [B, T, 4, 6, H, W]
        X_geo   = X[:, :, :, 7:, :, :]      # geo [B, T, 4, 2, H, W]

        # Extract wind components (last timestep)
        wind_u = X_meteo[:, -1, 0:1, :, :]
        wind_v = X_meteo[:, -1, 1:2, :, :]

        optimizer.zero_grad()

        outputs = model(
            X_fire=X_fire,
            X_meteo=X_meteo,
            X_geo=X_geo,
            wind_u=wind_u,
            wind_v=wind_v,
        )

        Y_pred = outputs["Y_final"]
        L_contrast = outputs["L_contrast"]
        losses = loss_fn(Y_pred, Y_true, outputs["embeddings"], L_contrast, wind_u, wind_v)
        total = losses["L_total"]

        total.backward()
        optimizer.step()
        total_loss += total.item()

    # ---- Epoch summary log ----
    avg_loss = total_loss / len(dataloader)
    metrics = test(model, dataloader, device, logger, region)  # 在每轮 epoch 结束后评估
    iou, auc, f1 = metrics["IoU"], metrics["AUC"], metrics["F1"]
    logger.info(
        f"[{region.upper()}] Epoch {epoch} | "
        f"Avg Loss={avg_loss:.4f} | IoU={iou:.4f} | AUC={auc:.4f} | F1={f1:.4f}"
    )
    return avg_loss


# ============================================================
# Testing Function
# ============================================================
def test(model, dataloader, device, logger, region):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X, Y_true in tqdm(dataloader, desc=f"[{region}] Testing"):
            X, Y_true = X.to(device), Y_true.to(device)

            # X_fire = X[:, :, 0:1, :, :]
            # X_meteo = X[:, :, 1:7, :, :]
            # X_geo = X[:, :, 7:, :, :]  # dynamic geo [B, T, 2, H, W]

            # # Input shape: [B, T, 4, 9, H, W]
            X_fire  = X[:, :, 0, 0:1, :, :]     # wildfire [B, T, 4, 1, H, W]
            X_meteo = X[:, :, :, 1:7, :, :]     # meteorology [B, T, 4, 6, H, W]
            X_geo   = X[:, :, :, 7:, :, :]      # geo [B, T, 4, 2, H, W]

            wind_u = X_meteo[:, -1, 0:1, :, :]
            wind_v = X_meteo[:, -1, 1:2, :, :]

            outputs = model(
                X_fire=X_fire,
                X_meteo=X_meteo,
                X_geo=X_geo,
                wind_u=wind_u,
                wind_v=wind_v,
            )

            preds.append(outputs["Y_final"].cpu())
            trues.append(Y_true.cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    metrics = compute_metrics(preds, trues)
    logger.info(f"[{region.upper()}] Test Results: {metrics}")
    return metrics


# ============================================================
# Main Entry
# ============================================================
def main():
    args = get_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    region = args.region or config["data"].get("region", "california")
    if region not in ["california", "florida"]:
        raise ValueError(f"❌ Invalid region '{region}'.")

    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.lr or config["training"]["lr"]
    device = torch.device(args.device)
    set_seed(config["training"].get("seed", 42))

    os.makedirs(config["logging"]["save_dir"], exist_ok=True)
    logger = get_logger(f"FireTrend_{region.upper()}", config["logging"]["save_dir"])
    logger.info(f"FireTrend started on region: {region.upper()}")

    dataloader = create_dataloader(
        data_root=config["data"]["root_dir"],
        region=region,
        seq_length=config["data"]["seq_length"],
        pred_horizon=config["data"]["pred_horizon"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        normalize=True,
    )
    
    # Dynamically assign spatial dimensions by region
    if region.lower() == "california":
        height, width = 49, 53
    elif region.lower() == "florida":
        height, width = 64, 64  # ⚠️ 替换成佛州真实维度
    else:
        raise ValueError(f"Unknown region '{region}' — please specify correct H,W.")

    logger.info(f"[Config] Region={region.upper()} | Input grid size=({height}, {width})")

    # Correct input dimensions for dynamic geo
    model = FireTrendModel(
        in_dims={"fire": 1, "meteo": 6, "geo": 2},
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        hidden_dim=config["model"]["hidden_dim"],
        height=config["model"]["height"],
        width=config["model"]["width"],
        kernel_size=config["model"]["kernel_size"],
        beta=config["model"]["beta"],
        verbose=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = FireTrendLoss().to(device)

    # ========================================================
    # Train / Test
    # ========================================================
    if args.train:
        best_loss = float("inf")
        for epoch in range(1, epochs + 1):
            avg_loss = train(model, dataloader, optimizer, loss_fn, epoch, device, logger, region)

            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, f"{region}_model_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"[{region.upper()}] Saved checkpoint: {ckpt_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.save_dir, f"{region}_model_best.pth")
                torch.save(model.state_dict(), best_path)
                logger.info(f"[{region.upper()}] New best model saved: {best_path}")

    elif args.test:
        if args.checkpoint is None or not os.path.exists(args.checkpoint):
            logger.error("❌ No valid checkpoint specified for testing.")
            return

        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
        metrics = test(model, dataloader, device, logger, region)
        logger.info(f"[{region.upper()}] Final Test Metrics: {metrics}")

    else:
        logger.warning("Please specify either --train or --test mode.")


if __name__ == "__main__":
    main()
