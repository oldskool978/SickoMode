from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class LimiterConfig:
    sample_rate: int = 48000
    chunk_size: int = 960
    lookahead_samples: int = 144
    oversample_factor: int = 8
    true_peak_ceiling_db: float = -0.3
    mode: str = "psychoacoustic_celt"
    num_celt_bands: int = 21
    smoothness_weight: float = 0.05
    transient_weight: float = 1.0
    spectral_flatness_offset_tone: float = 14.5
    spectral_flatness_offset_noise: float = 5.5
    basilar_compression_exponent: float = 0.33
    ath_min_db: float = -10.0
    stereo_link: bool = True
    prepass_enabled: bool = True
    device: str = "cuda"

    @property
    def linear_ceiling(self) -> float:
        return float(10.0 ** (self.true_peak_ceiling_db / 20.0))

    @property
    def target_ceiling_linear(self) -> float:
        return float(10.0 ** (self.true_peak_ceiling_db / 20.0))


@dataclass
class LimiterState:
    audio_buffer: Optional[torch.Tensor] = None
    envelope_buffer: Optional[torch.Tensor] = None
    filterbank_state: Optional[torch.Tensor] = None
    resampler_state: Optional[torch.Tensor] = None
    transient_energy_prev: Optional[torch.Tensor] = None
    prev_gains: Optional[torch.Tensor] = None
    prev_a: float = 0.0

    def reset(self, channels: int, config: LimiterConfig, device: torch.device):
        total_delay = config.lookahead_samples + 16  # Unified 160 samples system delay
        self.audio_buffer = torch.zeros((channels, total_delay), dtype=torch.float32, device=device)
        self.envelope_buffer = torch.zeros((1, 1, config.lookahead_samples), dtype=torch.float32, device=device)
        self.filterbank_state = torch.zeros((channels, 64), dtype=torch.float32, device=device)
        self.resampler_state = torch.zeros((channels, 32), dtype=torch.float32, device=device)
        num_trans_bands = max(1, config.num_celt_bands - 7) if config.num_celt_bands > 7 else config.num_celt_bands
        self.transient_energy_prev = torch.zeros((1, num_trans_bands, 1), dtype=torch.float32, device=device)
        num_gains = config.num_celt_bands if config.mode == "psychoacoustic_celt" else 1
        self.prev_gains = torch.ones((1, num_gains, 1), dtype=torch.float32, device=device)
        self.prev_a = 0.0


@dataclass
class AuditResult:
    filename: str
    mode: str
    passband_residual_linf: float
    input_true_peak_dbtp: float
    output_true_peak_dbtp: float
    input_peak_dbfs: float
    output_peak_dbfs: float
    max_gain_reduction_db: float
    stereo_correlation_in: float
    stereo_correlation_out: float
    stereo_phase_delta: float
    transient_smear_index: float
    calibrated_delay_samples: int
    celt_band_max_nmr_db: List[float]
    gain_acceleration_c2_max: float
    inter_sample_clip_count: int
    is_bit_exact_passband: bool
    is_true_peak_compliant: bool
    is_psychoacoustically_transparent: bool