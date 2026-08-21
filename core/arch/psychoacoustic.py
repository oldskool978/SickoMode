import torch


class PsychoacousticModel:
    def __init__(self, num_bands: int = 21, sample_rate: int = 48000, device: torch.device = torch.device("cpu")):
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.device = device
        self.ath = self._calculate_terhardt_ath()
        self.spreading_matrix = self._calculate_spreading_matrix()

    def _calculate_terhardt_ath(self) -> torch.Tensor:
        center_freqs = torch.tensor([
            100.0, 300.0, 500.0, 700.0, 900.0, 1100.0, 1300.0, 1500.0, 1800.0,
            2200.0, 2600.0, 3000.0, 3600.0, 4400.0, 5200.0, 6200.0, 7400.0,
            8800.0, 10800.0, 13800.0, 17800.0
        ], dtype=torch.float32, device=self.device)
        f_khz = torch.clamp(center_freqs / 1000.0, min=0.01)
        ath_db = (
            3.64 * torch.pow(f_khz, -0.8)
            - 6.5 * torch.exp(-0.6 * torch.pow(f_khz - 3.3, 2.0))
            + 1e-3 * torch.pow(f_khz, 4.0)
        )
        return torch.pow(10.0, ath_db / 10.0)

    def _calculate_spreading_matrix(self) -> torch.Tensor:
        barks = torch.linspace(1.0, 24.0, self.num_bands, device=self.device)
        s_matrix = torch.zeros((self.num_bands, self.num_bands), dtype=torch.float32, device=self.device)
        for i in range(self.num_bands):
            for j in range(self.num_bands):
                dz = barks[i] - barks[j]
                if dz >= 0.0:
                    val_db = -27.0 * dz
                else:
                    val_db = 24.0 * dz
                s_matrix[i, j] = 10.0 ** (val_db / 10.0)
        return s_matrix

    def compute_masking_thresholds(self, band_energies: torch.Tensor, sfm: torch.Tensor) -> torch.Tensor:
        combined_energy = torch.mean(band_energies, dim=0, keepdim=True)
        compressed_e = torch.pow(torch.clamp(combined_energy, min=1e-12), 0.33)
        spread_e = torch.matmul(compressed_e, self.spreading_matrix.T)
        excitation = torch.pow(torch.clamp(spread_e, min=1e-12), 3.0)

        sfm_mean = torch.mean(sfm, dim=0, keepdim=True)
        offset_db = sfm_mean * 5.5 + (1.0 - sfm_mean) * 14.5
        offset_linear = torch.pow(10.0, -offset_db / 10.0)

        thresholds = excitation * offset_linear
        thresholds = torch.maximum(thresholds, self.ath.unsqueeze(0))
        return thresholds