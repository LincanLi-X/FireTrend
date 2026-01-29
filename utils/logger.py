import os
import sys
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:
    """
    TensorBoard logger wrapper for FireTrend.
    """
    def __init__(self, log_dir="./outputs/logs"):
        os.makedirs(log_dir, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f"run_{run_id}"))

    def log_metrics(self, metrics_dict, step):
        """
        Log scalar metrics to TensorBoard.
        """
        for key, val in metrics_dict.items():
            if val is not None:
                self.writer.add_scalar(key, val, step)

    def log_images(self, tag, images, step):
        """
        Log spatial maps (e.g., fire risk maps).
        images: Tensor [B, 1, H, W] or [B, H, W]
        """
        if images.ndim == 3:
            images = images.unsqueeze(1)
        self.writer.add_images(tag, images, step)

    def close(self):
        self.writer.close()


def get_logger(name="FireTrend", log_dir="./outputs/logs", log_level=logging.INFO):
    """
    Create a combined console + file logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_format = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_format)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Logs saved to {log_path}")
    return logger
