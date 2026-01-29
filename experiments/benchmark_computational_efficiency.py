import time
import torch
import torch.nn as nn

# ============================================================
# Import PyroCast DirectionalConv2D (原版实现)
# ============================================================
from pyrocast_physics import DirectionalConv2D


# ============================================================
# Benchmark Utility
# ============================================================
@torch.no_grad()
def measure_runtime(model, inputs, device="cuda", repeat=30, warmup=10):
    """
    Measure average forward runtime (ms) and peak GPU memory (MB)
    """
    model = model.to(device)
    inputs = [x.to(device) for x in inputs]

    # Warmup
    for _ in range(warmup):
        _ = model(*inputs)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    for _ in range(repeat):
        _ = model(*inputs)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / repeat * 1000.0
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    return avg_time_ms, peak_mem_mb


# ============================================================
# Baseline: Standard Conv2D Wrapper
# ============================================================
class StandardConvWrapper(nn.Module):
    """
    Wrapper to align interface with DirectionalConv2D
    """
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, Y_pred, wind_u, wind_v):
        # wind inputs are ignored, kept only for interface consistency
        return self.conv(Y_pred)


# ============================================================
# Main Benchmark
# ============================================================
def run_benchmark():
    device = "cuda"
    #torch.manual_seed(0)

    batch_size = 8
    kernel_size = 5
    spatial_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]

    print("=" * 80)
    print("Computational Efficiency: Directional Conv vs Standard Conv")
    print("=" * 80)
    print("B\tH\tW\tStdConv(ms)\tDirConv(ms)\tStdMem(MB)\tDirMem(MB)")

    for H, W in spatial_sizes:
        # --- synthetic inputs ---
        Y_pred = torch.rand(batch_size, 1, H, W)
        wind_u = torch.rand(batch_size, 1, H, W)
        wind_v = torch.rand(batch_size, 1, H, W)

        # --- models ---
        std_conv = StandardConvWrapper(kernel_size=kernel_size)
        dir_conv = DirectionalConv2D(kernel_size=kernel_size, verbose=False)

        # --- measure ---
        std_t, std_m = measure_runtime(
            std_conv,
            inputs=[Y_pred, wind_u, wind_v],
            device=device,
        )

        dir_t, dir_m = measure_runtime(
            dir_conv,
            inputs=[Y_pred, wind_u, wind_v],
            device=device,
        )

        print(
            f"{batch_size}\t{H}\t{W}\t"
            f"{std_t:.3f}\t\t"
            f"{dir_t:.3f}\t\t"
            f"{std_m:.1f}\t\t"
            f"{dir_m:.1f}"
        )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for this benchmark."
    run_benchmark()
