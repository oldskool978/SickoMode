import torch
import torch.nn.functional as F


@torch.jit.script
def _fast_release_filter_cpu(
    a_attack: torch.Tensor,
    init_a: float,
    release_coeff: float
) -> torch.Tensor:
    """
    Executes 1-pole asymmetric release recursion in native CPU cache registers.
    Eliminates GPU global memory latency stalls on serial recurrences.
    """
    length = a_attack.numel()
    out = torch.empty_like(a_attack)
    curr_a = init_a
    for t in range(length):
        target = float(a_attack[t])
        if target >= curr_a:
            curr_a = target
        else:
            curr_a = target + (curr_a - target) * release_coeff
        out[t] = curr_a
    return out


class VariationalGainSolver:
    def __init__(
        self,
        num_bands: int = 21,
        lookahead: int = 144,
        sample_rate: int = 48000,
        release_time_sec: float = 0.025,
        device: torch.device = torch.device("cpu")
    ):
        self.num_bands = num_bands
        self.lookahead = lookahead
        self.device = device
        self.sample_rate = sample_rate
        self.release_coeff = float(torch.exp(torch.tensor(-1.0 / (sample_rate * release_time_sec))).item())
        self.psi_kernel = self._design_quintic_lookahead_kernel()

    def _design_quintic_lookahead_kernel(self) -> torch.Tensor:
        """Designs a 4D C2-continuous quintic smoothstep kernel: psi(u) = 1 - 10u^3 + 15u^4 - 6u^5."""
        u = torch.linspace(0.0, 1.0, self.lookahead + 1, dtype=torch.float32, device=self.device)
        psi = 1.0 - (10.0 * (u ** 3) - 15.0 * (u ** 4) + 6.0 * (u ** 5))
        return psi.view(1, 1, 1, -1)

    def solve_fir_streaming(
        self,
        tp_envelope: torch.Tensor,
        envelope_history: torch.Tensor,
        length: int,
        ceiling: float,
        prev_a: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Stateful streaming FIR lookahead with continuous boundary buffering.
        """
        effective_ceiling = ceiling * (10.0 ** (-0.02 / 20.0))
        full_env = torch.cat([envelope_history, tp_envelope], dim=-1)
        new_env_history = full_env[..., -self.lookahead:]

        g_bound = torch.clamp(effective_ceiling / (full_env + 1e-12), max=1.0)
        a_bound = 1.0 - g_bound
        unfolded = a_bound.unfold(dimension=-1, size=self.lookahead + 1, step=1)
        scaled = unfolded * self.psi_kernel
        a_attack = torch.max(scaled, dim=-1).values[..., :length]

        a_attack_cpu = a_attack.view(-1).cpu().contiguous()
        a_smoothed_cpu = _fast_release_filter_cpu(a_attack_cpu, prev_a, self.release_coeff)
        last_a = float(a_smoothed_cpu[-1].item()) if a_smoothed_cpu.numel() > 0 else 0.0

        a_smoothed = a_smoothed_cpu.to(self.device).view(1, 1, length)
        g_out = torch.clamp(1.0 - a_smoothed, min=1e-4, max=1.0)
        return g_out, new_env_history, last_a

    def solve_fir_linear_phase(
        self,
        tp_envelope: torch.Tensor,
        length: int,
        ceiling: float,
        prev_gain: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        effective_ceiling = ceiling * (10.0 ** (-0.02 / 20.0))
        g_bound = torch.clamp(effective_ceiling / (tp_envelope + 1e-12), max=1.0)
        a_bound = 1.0 - g_bound
        a_padded = F.pad(a_bound, (0, self.lookahead), mode="replicate")
        unfolded = a_padded.unfold(dimension=-1, size=self.lookahead + 1, step=1)
        scaled = unfolded * self.psi_kernel
        a_attack = torch.max(scaled, dim=-1).values[..., :length]

        a_attack_cpu = a_attack.view(-1).cpu().contiguous()
        init_a = float(1.0 - prev_gain.min().item()) if prev_gain is not None else 0.0
        a_smoothed_cpu = _fast_release_filter_cpu(a_attack_cpu, init_a, self.release_coeff)

        a_smoothed = a_smoothed_cpu.to(self.device).view(1, 1, length)
        g_out = torch.clamp(1.0 - a_smoothed, min=1e-4, max=1.0)
        new_prev = g_out[:, :, -1:]
        return g_out, new_prev

    def solve_psychoacoustic_celt_streaming(
        self,
        tp_envelope: torch.Tensor,
        envelope_history: torch.Tensor,
        length: int,
        subbands: torch.Tensor,
        transient_mask: torch.Tensor,
        masking_thresholds: torch.Tensor,
        ceiling: float,
        prev_a: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Stateful streaming CELT multiband solver with phase-locked master envelope.
        """
        effective_ceiling = ceiling * (10.0 ** (-0.02 / 20.0))
        g_master, new_env_history, last_a = self.solve_fir_streaming(
            tp_envelope=tp_envelope,
            envelope_history=envelope_history,
            length=length,
            ceiling=ceiling,
            prev_a=prev_a
        )

        band_powers = torch.mean(subbands ** 2, dim=-1, keepdim=True) + 1e-12
        combined_powers = torch.mean(band_powers, dim=0, keepdim=True)
        mask_weights = masking_thresholds.unsqueeze(-1) + 1e-12
        sensitivity = combined_powers / mask_weights
        sensitivity_norm = sensitivity / (torch.max(sensitivity, dim=1, keepdim=True).values + 1e-12)

        w_stat = 1.0 - 0.15 * sensitivity_norm * (1.0 - g_master)
        w_trans = torch.ones_like(g_master)
        w_final = (1.0 - transient_mask) * w_stat + transient_mask * w_trans
        g_bands = g_master * w_final

        scaled_subbands = g_bands * subbands
        sum_scaled = torch.sum(scaled_subbands, dim=1)
        max_sum = torch.max(torch.abs(sum_scaled), dim=0, keepdim=True).values.unsqueeze(0)
        overshoot = torch.clamp(max_sum / effective_ceiling, min=1.0)
        g_final = torch.clamp(g_bands / overshoot, min=1e-4, max=1.0)
        return g_final, new_env_history, last_a

    def solve_psychoacoustic_celt(
        self,
        tp_envelope: torch.Tensor,
        length: int,
        subbands: torch.Tensor,
        transient_mask: torch.Tensor,
        masking_thresholds: torch.Tensor,
        ceiling: float,
        prev_gains: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g_master, _ = self.solve_fir_linear_phase(
            tp_envelope=tp_envelope,
            length=length,
            ceiling=ceiling,
            prev_gain=prev_gains
        )

        band_powers = torch.mean(subbands ** 2, dim=-1, keepdim=True) + 1e-12
        combined_powers = torch.mean(band_powers, dim=0, keepdim=True)
        mask_weights = masking_thresholds.unsqueeze(-1) + 1e-12
        sensitivity = combined_powers / mask_weights
        sensitivity_norm = sensitivity / (torch.max(sensitivity, dim=1, keepdim=True).values + 1e-12)

        w_stat = 1.0 - 0.15 * sensitivity_norm * (1.0 - g_master)
        w_trans = torch.ones_like(g_master)
        w_final = (1.0 - transient_mask) * w_stat + transient_mask * w_trans
        g_bands = g_master * w_final

        scaled_subbands = g_bands * subbands
        sum_scaled = torch.sum(scaled_subbands, dim=1)
        max_sum = torch.max(torch.abs(sum_scaled), dim=0, keepdim=True).values.unsqueeze(0)
        overshoot = torch.clamp(max_sum / ceiling, min=1.0)
        g_final = torch.clamp(g_bands / overshoot, min=1e-4, max=1.0)
        new_prev = g_final[:, :, -1:]
        return g_final, new_prev

    def solve_prepass_spline(
        self,
        full_audio: torch.Tensor,
        resampler,
        ceiling: float
    ) -> torch.Tensor:
        channels, length = full_audio.shape
        chunk_size = 48000

        # Pass 1: Forward predictive lookahead attenuation
        g_chunks = []
        prev_g = None
        state_tp = torch.zeros((channels, resampler.pad_samples), dtype=torch.float32, device=self.device)
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            chunk = full_audio[:, start:end]
            c_len = end - start
            tp_env, state_tp = resampler.extract_true_peak_streaming(chunk, state_tp)
            g_c, prev_g = self.solve_fir_linear_phase(
                tp_envelope=tp_env,
                length=c_len,
                ceiling=ceiling,
                prev_gain=prev_g
            )
            g_chunks.append(g_c)
        g_global = torch.cat(g_chunks, dim=-1)

        # Pass 2: Localized Intersample Verification & Secondary Lookahead Pass
        y_trial = g_global.squeeze(0) * full_audio
        post_tp_env = resampler.extract_true_peak_envelope(y_trial)
        max_post_tp = float(torch.max(post_tp_env).item())

        if max_post_tp > ceiling:
            g2_chunks = []
            prev_g2 = None
            state_tp2 = torch.zeros((channels, resampler.pad_samples), dtype=torch.float32, device=self.device)
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                c_len = end - start
                chunk_post_tp, state_tp2 = resampler.extract_true_peak_streaming(y_trial[:, start:end], state_tp2)
                g_c2, prev_g2 = self.solve_fir_linear_phase(
                    tp_envelope=chunk_post_tp,
                    length=c_len,
                    ceiling=ceiling,
                    prev_gain=prev_g2
                )
                g2_chunks.append(g_c2)
            g_secondary = torch.cat(g2_chunks, dim=-1)
            g_global = torch.clamp(g_global * g_secondary, min=1e-4, max=1.0)

        return g_global