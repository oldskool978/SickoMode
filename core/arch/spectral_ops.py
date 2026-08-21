import torch
import torch.nn.functional as F
from core.schema import LimiterConfig
from core.dsp_cuda import design_polyphase_kaiser_bank

class TruePeakPolyphaseResampler:
    def __init__(self, config: LimiterConfig, device: torch.device):
        self.factor = config.oversample_factor
        self.device = device
        self.half_length = 16
        self.poly_bank = design_polyphase_kaiser_bank(
            oversample_factor=self.factor,
            half_length=self.half_length,
            beta=9.5,
            device=device
        )
        self.pad_samples = self.half_length * 2

    @torch.inference_mode()
    def extract_true_peak_streaming(
        self,
        x: torch.Tensor,
        state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channels, length = x.shape
        full_x = torch.cat([state, x], dim=-1)
        new_state = full_x[:, -self.pad_samples:]

        x_reshaped = full_x.view(channels, 1, full_x.shape[-1])
        x_branches = F.conv1d(x_reshaped, self.poly_bank, padding=0)
        x_branches_valid = x_branches[:, :, :length]
        tp_env = torch.max(torch.abs(x_branches_valid), dim=1).values
        spatial_tp = torch.max(tp_env, dim=0, keepdim=True).values.unsqueeze(0)
        return spatial_tp, new_state

    @torch.inference_mode()
    def extract_true_peak_envelope(self, x: torch.Tensor) -> torch.Tensor:
        channels, length = x.shape
        x_padded = F.pad(x, (self.half_length, self.half_length), mode="reflect")
        x_reshaped = x_padded.view(channels, 1, x_padded.shape[-1])
        x_branches = F.conv1d(x_reshaped, self.poly_bank, padding=0)
        x_branches_valid = x_branches[:, :, :length]
        tp_env = torch.max(torch.abs(x_branches_valid), dim=1).values
        spatial_tp = torch.max(tp_env, dim=0, keepdim=True).values.unsqueeze(0)
        return spatial_tp

    @torch.inference_mode()
    def measure_true_peak(self, x: torch.Tensor) -> float:
        spatial_tp = self.extract_true_peak_envelope(x)
        max_peak = torch.max(spatial_tp).item()
        return float(20.0 * torch.log10(torch.tensor(max(max_peak, 1e-12))).item())