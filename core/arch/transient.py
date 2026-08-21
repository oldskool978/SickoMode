import torch


class TransientDeGroupingEngine:
    def __init__(self, sample_rate: int = 48000, device: torch.device = torch.device("cpu")):
        self.sample_rate = sample_rate
        self.device = device
        self.alpha_flux = 0.95
        self.split_band = 7

    def detect_transient_mask_streaming(
        self,
        subbands: torch.Tensor,
        prev_energy: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channels, num_bands, length = subbands.shape
        start_b = min(self.split_band, num_bands - 1)
        energy = torch.mean(subbands[:, start_b:] ** 2, dim=0, keepdim=True)

        if prev_energy is None or prev_energy.shape[1] != energy.shape[1] or prev_energy.device != energy.device:
            prev_energy = torch.zeros((1, energy.shape[1], 1), dtype=energy.dtype, device=energy.device)

        energy_full = torch.cat([prev_energy, energy], dim=-1)
        new_prev = energy_full[:, :, -1:]

        diff_energy = torch.diff(energy_full, dim=-1)
        pos_flux = torch.clamp(diff_energy, min=0.0)
        spectral_flux = torch.sum(pos_flux, dim=1, keepdim=True)
        mean_flux = torch.mean(spectral_flux, dim=-1, keepdim=True) + 1e-9

        transient_weight = torch.sigmoid((spectral_flux - 2.0 * mean_flux) / (mean_flux + 1e-6))
        return transient_weight, new_prev

    def detect_transient_mask(self, subbands: torch.Tensor) -> torch.Tensor:
        channels, num_bands, length = subbands.shape
        start_b = min(self.split_band, num_bands - 1)
        energy = torch.mean(subbands[:, start_b:] ** 2, dim=0, keepdim=True)
        diff_energy = torch.diff(energy, dim=-1, prepend=energy[..., :1])
        pos_flux = torch.clamp(diff_energy, min=0.0)
        spectral_flux = torch.sum(pos_flux, dim=1, keepdim=True)
        mean_flux = torch.mean(spectral_flux, dim=-1, keepdim=True) + 1e-9
        transient_weight = torch.sigmoid((spectral_flux - 2.0 * mean_flux) / (mean_flux + 1e-6))
        return transient_weight