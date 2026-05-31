"""
FireTrend training and evaluation entrypoint.

The default training mode follows the new manuscript:
    pretrain label-free representations, then fine-tune the ordinal classifier.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from modules.firetrend_model import FireTrendModel
from modules.losses import FireTrendLoss
from utils.data_loader import create_dataloader
from utils.logger import get_logger
from utils.metrics import compute_metrics
from utils.seed_utils import set_seed


def get_args():
    parser = argparse.ArgumentParser(description="FireTrend")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrain_epochs", type=int, default=None)
    parser.add_argument("--finetune_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["pretrain", "finetune", "joint", "pretrain_then_finetune"],
    )
    parser.add_argument("--region", type=str, default=None, choices=["california", "florida", "ca", "fl"])
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_path(path: str | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    return str(path_obj if path_obj.is_absolute() else base_dir / path_obj)


def as_float(value, default=None):
    if value is None:
        return default
    return float(value)


def unpack_batch(batch, device: torch.device):
    if isinstance(batch, dict):
        X = batch["x"].to(device, non_blocking=True)
        y_class = batch["y_class"].to(device, non_blocking=True)
        drivers = batch.get("drivers")
        drivers = drivers.to(device, non_blocking=True) if drivers is not None else None
    elif isinstance(batch, (tuple, list)) and len(batch) >= 4:
        X, y_class, _, drivers = batch[:4]
        X = X.to(device, non_blocking=True)
        y_class = y_class.to(device, non_blocking=True)
        drivers = drivers.to(device, non_blocking=True)
    else:
        raise ValueError("Expected a FireDataset dict batch or a 4-tuple batch.")

    X_fire = X[:, :, 0:1]
    X_meteo = X[:, :, 1:9]
    X_geo = X[:, :, 9:19]
    return X_fire, X_meteo, X_geo, drivers, y_class


def model_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def load_checkpoint(model, checkpoint_path: str, device: torch.device, logger):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    target = model.module if hasattr(model, "module") else model
    missing, unexpected = target.load_state_dict(state, strict=False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    if missing:
        logger.warning(f"Missing checkpoint keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected checkpoint keys: {unexpected}")


def set_trainable_for_stage(model, stage: str, freeze_encoder: bool) -> None:
    target = model.module if hasattr(model, "module") else model
    for param in target.parameters():
        param.requires_grad = True
    if stage == "finetune" and freeze_encoder:
        for module in [target.encoder, target.contrastive, target.pyrocast, target.latent_fusion]:
            for param in module.parameters():
                param.requires_grad = False


def run_epoch(model, dataloader, optimizer, loss_fn, device, stage, logger, region, epoch):
    model.train()
    set_trainable_for_stage(
        model,
        stage=stage,
        freeze_encoder=getattr(run_epoch, "freeze_encoder", False),
    )
    totals = {}
    progress = tqdm(dataloader, desc=f"[{region}] {stage} epoch {epoch}")

    for batch in progress:
        X_fire, X_meteo, X_geo, drivers, y_class = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            X_fire=X_fire,
            X_meteo=X_meteo,
            X_geo=X_geo,
            X_drivers=drivers,
            compute_pretrain=stage in {"pretrain", "joint"},
        )
        losses = loss_fn(outputs, None if stage == "pretrain" else y_class, stage=stage)
        losses["L_total"].backward()
        optimizer.step()

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        progress.set_postfix(loss=f"{float(losses['L_total'].detach().cpu()):.4f}")

    avg = {key: value / max(len(dataloader), 1) for key, value in totals.items()}
    logger.info(
        f"[{region.upper()}] {stage} epoch {epoch} | "
        + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
    )
    return avg


@torch.no_grad()
def evaluate(model, dataloader, device, logger, region):
    model.eval()
    logits_all, targets_all = [], []
    for batch in tqdm(dataloader, desc=f"[{region}] evaluation"):
        X_fire, X_meteo, X_geo, drivers, y_class = unpack_batch(batch, device)
        outputs = model(
            X_fire=X_fire,
            X_meteo=X_meteo,
            X_geo=X_geo,
            X_drivers=drivers,
            compute_pretrain=False,
        )
        logits_all.append(outputs["logits"].detach().cpu())
        targets_all.append(y_class.detach().cpu())

    logits = torch.cat(logits_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    metrics = compute_metrics(logits, targets, num_classes=logits.size(1))
    logger.info(f"[{region.upper()}] evaluation | {metrics}")
    return metrics


def save_checkpoint(model, path: str, stage: str, epoch: int, config: dict, logger) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model_state_dict(model),
            "stage": stage,
            "epoch": epoch,
            "config": config,
        },
        path,
    )
    logger.info(f"Saved checkpoint: {path}")


def build_dataloader(config, region, batch_size, split, shuffle, config_dir):
    data_cfg = config["data"]
    return create_dataloader(
        data_root=resolve_path(data_cfg["root_dir"], config_dir),
        region=region,
        seq_length=int(data_cfg.get("seq_length", 8)),
        pred_horizon=int(data_cfg.get("pred_horizon", 1)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", data_cfg.get("num_workers", 0))),
        normalize=bool(data_cfg.get("normalize", True)),
        risk_thresholds=data_cfg.get("risk_thresholds", [33.3333, 66.6667]),
        split=split,
        split_ratios=data_cfg.get("split_ratios", [0.7, 0.1, 0.2]),
        return_dict=True,
    )


def build_model(config, height, width, device):
    model_cfg = config["model"]
    contrast_cfg = config.get("contrastive", {})
    pyro_cfg = config.get("pyrocast", {})
    model = FireTrendModel(
        in_dims={"fire": 1, "meteo": 8, "geo": 10},
        embed_dim=int(model_cfg.get("embed_dim", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        height=height,
        width=width,
        num_layers=int(model_cfg.get("num_layers", 1)),
        num_classes=int(model_cfg.get("num_classes", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        kernel_size=int(pyro_cfg.get("kernel_size", model_cfg.get("kernel_size", 5))),
        pyro_rho=float(pyro_cfg.get("rho", 0.35)),
        pyro_sigma_parallel=float(pyro_cfg.get("sigma_parallel", 1.5)),
        pyro_sigma_perp=float(pyro_cfg.get("sigma_perp", 0.75)),
        temperature=float(contrast_cfg.get("temperature", 0.07)),
        lambda_temporal=float(contrast_cfg.get("lambda_temporal", 1.0)),
        lambda_spatial=float(contrast_cfg.get("lambda_spatial", 1.0)),
        lambda_cross=float(contrast_cfg.get("lambda_cross", 1.0)),
        max_temporal_cells=int(contrast_cfg.get("max_temporal_cells", 512)),
        max_spatial_anchors=int(contrast_cfg.get("max_spatial_anchors", 128)),
        max_cross_samples=int(contrast_cfg.get("max_cross_samples", 1024)),
        meteo_driver_indices=model_cfg.get("meteo_driver_indices", None),
        detach_pyro_target=bool(pyro_cfg.get("detach_target", True)),
        verbose=bool(config.get("debug", {}).get("verbose", False)),
    ).to(device)

    if (
        device.type == "cuda"
        and torch.cuda.device_count() > 1
        and bool(config["training"].get("data_parallel", True))
    ):
        model = torch.nn.DataParallel(model)
    return model


def main():
    args = get_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    region = (args.region or config["data"].get("region", "california")).lower()
    if region == "ca":
        region = "california"
    if region == "fl":
        region = "florida"
    if region not in {"california", "florida"}:
        raise ValueError("Only FireCast-CA and FireCast-FL are enabled in this code release.")

    train_cfg = config["training"]
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 2))
    lr = as_float(args.lr, as_float(train_cfg.get("lr", 1e-4)))
    weight_decay = as_float(train_cfg.get("weight_decay", 0.0))
    stage = args.stage or train_cfg.get("stage", "pretrain_then_finetune")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() and config["device"].get("use_gpu", True) else "cpu"))

    set_seed(int(train_cfg.get("seed", 42)), deterministic=bool(train_cfg.get("deterministic", False)))

    log_dir = resolve_path(config["logging"]["save_dir"], config_dir)
    save_dir = args.save_dir or config["outputs"].get("checkpoint_dir", "./outputs/checkpoints")
    save_dir = resolve_path(save_dir, config_dir)
    logger = get_logger(f"FireTrend_{region.upper()}", log_dir)
    logger.info(f"FireTrend started | region={region} | stage={stage} | device={device}")

    train_loader = build_dataloader(
        config,
        region=region,
        batch_size=batch_size,
        split=config["data"].get("train_split", "train"),
        shuffle=True,
        config_dir=config_dir,
    )
    eval_loader = build_dataloader(
        config,
        region=region,
        batch_size=batch_size,
        split=config["data"].get("eval_split", "val"),
        shuffle=False,
        config_dir=config_dir,
    )

    height, width = train_loader.dataset.height, train_loader.dataset.width
    logger.info(f"Grid size=({height}, {width}) | class_counts={train_loader.dataset.class_counts.tolist()}")

    model = build_model(config, height, width, device)
    class_weights = train_loader.dataset.class_weights.to(device)
    loss_fn = FireTrendLoss(
        class_weights=class_weights,
        lambda_pyro=float(config.get("pyrocast", {}).get("lambda_pyro", 1.0)),
        lambda_cls=float(train_cfg.get("lambda_cls", 1.0)),
    ).to(device)
    logger.info(f"Class weights={class_weights.detach().cpu().tolist()}")

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device, logger)

    if args.test:
        if args.checkpoint is None:
            logger.warning("Testing without an explicit checkpoint uses the current randomly initialized model.")
        evaluate(model, eval_loader, device, logger, region)
        return

    if not args.train:
        logger.warning("Please specify --train or --test.")
        return

    run_epoch.freeze_encoder = bool(train_cfg.get("freeze_encoder_during_finetune", False))

    if stage == "pretrain_then_finetune":
        stages = [
            ("pretrain", int(args.pretrain_epochs or train_cfg.get("pretrain_epochs", 20))),
            ("finetune", int(args.finetune_epochs or train_cfg.get("finetune_epochs", 60))),
        ]
    else:
        epochs = int(args.epochs or train_cfg.get("epochs", 80))
        stages = [(stage, epochs)]

    for stage_name, n_epochs in stages:
        set_trainable_for_stage(
            model,
            stage=stage_name,
            freeze_encoder=bool(train_cfg.get("freeze_encoder_during_finetune", False)),
        )
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        best_iou = -1.0
        for epoch in range(1, n_epochs + 1):
            run_epoch(model, train_loader, optimizer, loss_fn, device, stage_name, logger, region, epoch)
            metrics = evaluate(model, eval_loader, device, logger, region)
            ckpt_path = os.path.join(save_dir, f"{region}_{stage_name}_epoch_{epoch:03d}.pth")
            save_checkpoint(model, ckpt_path, stage_name, epoch, config, logger)
            if metrics["IoU"] > best_iou:
                best_iou = metrics["IoU"]
                best_path = os.path.join(save_dir, f"{region}_{stage_name}_best.pth")
                save_checkpoint(model, best_path, stage_name, epoch, config, logger)


if __name__ == "__main__":
    main()
