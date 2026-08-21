import torch
import torch.nn.functional as F
from core.schema import LimiterConfig, LimiterState
from core.dsp_cuda import resolve_device
from core.arch.spectral_ops import TruePeakPolyphaseResampler
from core.arch.transient import TransientDeGroupingEngine
from core.arch.filterbank import CeltCriticalFilterbank
from core.arch.psychoacoustic import PsychoacousticModel
from core.arch.solver import VariationalGainSolver


class PsychoacousticLimiterEngine:
    def __init__(self, config: LimiterConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.resampler = TruePeakPolyphaseResampler(config, self.device)
        self.transient_engine = TransientDeGroupingEngine(sample_rate=config.sample_rate, device=self.device)
        self.filterbank = CeltCriticalFilterbank(sample_rate=config.sample_rate, device=self.device)
        self.psycho_model = PsychoacousticModel(num_bands=config.num_celt_bands, sample_rate=config.sample_rate, device=self.device)
        self.solver = VariationalGainSolver(
            num_bands=config.num_celt_bands,
            lookahead=config.lookahead_samples,
            sample_rate=config.sample_rate,
            device=self.device
        )
        self.total_delay = config.lookahead_samples + 16

    @torch.inference_mode()
    def process_block(self, chunk: torch.Tensor, state: LimiterState) -> torch.Tensor:
        chunk = chunk.to(self.device)
        channels, length = chunk.shape
        ceiling = self.config.linear_ceiling
        if state.audio_buffer is None:
            state.reset(channels=channels, config=self.config, device=self.device)

        fb_tap = self.filterbank.center_tap  # 32 samples

        # 1. Manage unified audio delay line (Total delay = 160 samples)
        full_stream = torch.cat([state.audio_buffer, chunk], dim=-1)
        x_dry = full_stream[:, :length]
        state.audio_buffer = full_stream[:, length:]

        # 2. Extract True-Peak envelope from incoming chunk
        tp_envelope, state.resampler_state = self.resampler.extract_true_peak_streaming(
            chunk, state.resampler_state
        )

        if self.config.mode == "fir_linear_phase":
            # FIR streaming lookahead
            g_scalar, state.envelope_buffer, state.prev_a = self.solver.solve_fir_streaming(
                tp_envelope=tp_envelope,
                envelope_history=state.envelope_buffer,
                length=length,
                ceiling=ceiling,
                prev_a=state.prev_a
            )
            y_out = g_scalar.squeeze(0) * x_dry
        else:
            # CELT multiband analysis fed with 128-sample delayed stream
            # (128 delay + 32 filterbank center tap = exact 160 sample alignment with x_dry)
            x_fb_in = full_stream[:, fb_tap : fb_tap + length]
            subbands, state.filterbank_state = self.filterbank.analyze_streaming(
                x_fb_in, state.filterbank_state
            )
            transient_mask, state.transient_energy_prev = self.transient_engine.detect_transient_mask_streaming(
                subbands, state.transient_energy_prev
            )
            band_powers = torch.mean(subbands ** 2, dim=-1)
            log_powers = torch.log(torch.clamp(band_powers, min=1e-12))
            geom_mean = torch.exp(torch.mean(log_powers, dim=-1, keepdim=True))
            arith_mean = torch.mean(band_powers, dim=-1, keepdim=True) + 1e-12
            sfm = torch.clamp(geom_mean / arith_mean, min=0.0, max=1.0)
            masking_thresholds = self.psycho_model.compute_masking_thresholds(band_powers, sfm)

            g_bands, state.envelope_buffer, state.prev_a = self.solver.solve_psychoacoustic_celt_streaming(
                tp_envelope=tp_envelope,
                envelope_history=state.envelope_buffer,
                length=length,
                subbands=subbands,
                transient_mask=transient_mask,
                masking_thresholds=masking_thresholds,
                ceiling=ceiling,
                prev_a=state.prev_a
            )
            delta_mod = (1.0 - g_bands) * subbands
            delta_synth = self.filterbank.synthesize(delta_mod)
            y_out = x_dry - delta_synth

        return y_out

    @torch.inference_mode()
    def process_full_prepass(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.to(self.device)
        channels, length = audio.shape
        ceiling = self.config.linear_ceiling

        g_global = self.solver.solve_prepass_spline(audio, resampler=self.resampler, ceiling=ceiling)

        if self.config.mode == "fir_linear_phase":
            y_out = g_global.squeeze(0) * audio
        else:
            subbands = self.filterbank.analyze(audio)
            transient_mask = self.transient_engine.detect_transient_mask(subbands)

            band_powers = torch.mean(subbands ** 2, dim=-1)
            log_powers = torch.log(torch.clamp(band_powers, min=1e-12))
            geom_mean = torch.exp(torch.mean(log_powers, dim=-1, keepdim=True))
            arith_mean = torch.mean(band_powers, dim=-1, keepdim=True) + 1e-12
            sfm = torch.clamp(geom_mean / arith_mean, min=0.0, max=1.0)
            masking_thresholds = self.psycho_model.compute_masking_thresholds(band_powers, sfm)

            combined_powers = torch.mean(torch.mean(subbands ** 2, dim=-1, keepdim=True), dim=0, keepdim=True)
            sensitivity = combined_powers / (masking_thresholds.unsqueeze(-1) + 1e-12)
            sensitivity_norm = sensitivity / (torch.max(sensitivity, dim=1, keepdim=True).values + 1e-12)

            w_stat = 1.0 - 0.15 * sensitivity_norm * (1.0 - g_global)
            w_trans = torch.ones_like(g_global)
            w_final = (1.0 - transient_mask) * w_stat + transient_mask * w_trans
            g_bands = g_global * w_final

            scaled_subbands = g_bands * subbands
            sum_scaled = torch.sum(scaled_subbands, dim=1)
            max_sum = torch.max(torch.abs(sum_scaled), dim=0, keepdim=True).values.unsqueeze(0)
            overshoot = torch.clamp(max_sum / ceiling, min=1.0)
            g_final = torch.clamp(g_bands / overshoot, min=1e-4, max=1.0)

            delta_mod = (1.0 - g_final) * subbands
            delta_synth = self.filterbank.synthesize(delta_mod)
            y_out = audio - delta_synth

        return y_out