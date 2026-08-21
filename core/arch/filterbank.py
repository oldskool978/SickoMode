import torch
import torch.nn.functional as F

class CeltCriticalFilterbank:
    EBANDS_HZ = [
        0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400,
        2800, 3200, 4000, 4800, 5600, 6800, 8000, 9600, 12000, 15600, 20000
    ]

    def __init__(self, sample_rate: int = 48000, num_taps: int = 65, device: torch.device = torch.device("cpu")):
        self.sample_rate = sample_rate
        self.num_bands = len(self.EBANDS_HZ) - 1
        self.num_taps = num_taps if num_taps % 2 == 1 else num_taps + 1
        self.device = device
        self.center_tap = (self.num_taps - 1) // 2
        self.filters = self._design_partition_of_unity_filters()

    def _design_partition_of_unity_filters(self) -> torch.Tensor:
        nyquist = self.sample_rate / 2.0
        t = torch.arange(self.num_taps, dtype=torch.float32, device=self.device) - float(self.center_tap)
        window = torch.hann_window(self.num_taps, device=self.device)
        filters = []
        for i in range(self.num_bands):
            f_low = max(self.EBANDS_HZ[i] / nyquist, 0.0)
            f_high = min(self.EBANDS_HZ[i + 1] / nyquist, 1.0)
            sinc_high = torch.where(
                t == 0.0,
                torch.tensor(f_high, device=self.device),
                torch.sin(torch.pi * f_high * t) / (torch.pi * t)
            ) if f_high < 1.0 else torch.where(t == 0.0, torch.tensor(1.0, device=self.device), torch.zeros_like(t))
            
            sinc_low = torch.where(
                t == 0.0,
                torch.tensor(f_low, device=self.device),
                torch.sin(torch.pi * f_low * t) / (torch.pi * t)
            ) if f_low > 0.0 else torch.zeros_like(t)
            
            h = (sinc_high - sinc_low) * window
            filters.append(h)
        filter_stack = torch.stack(filters, dim=0)
        sum_filters = torch.sum(filter_stack, dim=0)
        delta_target = torch.zeros_like(sum_filters)
        delta_target[self.center_tap] = 1.0
        correction = (delta_target - sum_filters) / float(self.num_bands)
        filter_stack = filter_stack + correction.unsqueeze(0)
        return filter_stack.unsqueeze(1)

    def analyze_streaming(
        self,
        x: torch.Tensor,
        state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_x = torch.cat([state, x], dim=-1)
        new_state = full_x[:, -(self.num_taps - 1):]
        channels = x.shape[0]
        bands = F.conv1d(full_x.unsqueeze(1), self.filters, padding=0)
        return bands, new_state

    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.center_tap
        x_padded = F.pad(x, (pad, pad), mode="reflect")
        bands = F.conv1d(x_padded.unsqueeze(1), self.filters, padding=0)
        return bands

    def synthesize(self, bands: torch.Tensor) -> torch.Tensor:
        return torch.sum(bands, dim=1)