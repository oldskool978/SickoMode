import os
import torch
import torch.nn.functional as F

os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"
os.environ["HIP_FORCE_DEV_STREAM"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = False

def resolve_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def design_polyphase_kaiser_bank(
    oversample_factor: int = 8,
    half_length: int = 16,
    beta: float = 9.5,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    num_taps = 2 * half_length * oversample_factor + 1
    t = torch.linspace(-half_length, half_length, num_taps, dtype=torch.float32, device=device)
    sinc = torch.where(t == 0.0, torch.ones_like(t), torch.sin(torch.pi * t) / (torch.pi * t))
    n = torch.arange(num_taps, dtype=torch.float32, device=device)
    alpha = (num_taps - 1) / 2.0
    val = beta * torch.sqrt(torch.clamp(1.0 - ((n - alpha) / alpha) ** 2, min=0.0))
    window = torch.i0(val) / torch.i0(torch.tensor(beta, dtype=torch.float32, device=device))
    kernel = sinc * window
    kernel = kernel / torch.sum(kernel[::oversample_factor])

    subfilter_taps = 2 * half_length + 1
    poly_bank = torch.zeros((oversample_factor, 1, subfilter_taps), dtype=torch.float32, device=device)
    for k in range(oversample_factor):
        branch = kernel[k::oversample_factor]
        if branch.shape[-1] < subfilter_taps:
            branch = F.pad(branch, (0, subfilter_taps - branch.shape[-1]))
        elif branch.shape[-1] > subfilter_taps:
            branch = branch[:subfilter_taps]
        poly_bank[k, 0, :] = branch.flip(-1)

    return poly_bank