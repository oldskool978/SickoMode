import sys
import os
import gc

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import json
import glob
import time
import yaml
import soundfile as sf
import torch
import numpy as np

from core.schema import LimiterConfig, LimiterState, AuditResult
from core.engine import PsychoacousticLimiterEngine


def load_config(config_path: str) -> LimiterConfig:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return LimiterConfig(**cfg_dict)


def find_system_delay_lag(x_in: torch.Tensor, x_out: torch.Tensor, max_search: int = 400) -> int:
    sample_len = min(16000, x_in.shape[-1], x_out.shape[-1])
    sig_in = x_in[0, :sample_len].cpu().numpy()
    sig_out = x_out[0, :sample_len].cpu().numpy()
    corr = np.correlate(sig_out, sig_in, mode="full")
    mid = len(sig_in) - 1
    search_region = corr[mid : mid + max_search]
    best_lag = int(np.argmax(search_region))
    return best_lag


@torch.inference_mode()
def compute_empirical_nmr(
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    engine: PsychoacousticLimiterEngine,
    delay_samples: int = 0,
    chunk_size: int = 48000
) -> list[float]:
    total_len = min(x_in.shape[-1] - delay_samples, x_out.shape[-1] - delay_samples)
    if total_len <= 2:
        return [-120.0] * engine.config.num_celt_bands
    
    channels = x_in.shape[0]
    num_bands = engine.config.num_celt_bands
    
    total_in_power = torch.zeros((channels, num_bands), dtype=torch.float32, device=engine.device)
    total_err_power = torch.zeros((channels, num_bands), dtype=torch.float32, device=engine.device)
    total_samples = 0
    
    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)
        cur_len = end - start
        
        c_in = x_in[:, start : end].to(engine.device)
        c_out = x_out[:, start + delay_samples : end + delay_samples].to(engine.device)
        c_err = c_in - c_out
        
        bands_in = engine.filterbank.analyze(c_in)
        bands_err = engine.filterbank.analyze(c_err)
        
        total_in_power += torch.sum(bands_in ** 2, dim=-1)
        total_err_power += torch.sum(bands_err ** 2, dim=-1)
        total_samples += cur_len

    in_power = (total_in_power / float(total_samples)) + 1e-12
    err_power = (total_err_power / float(total_samples))
    
    sfm = torch.ones((channels, 1), device=engine.device) * 0.5
    thresholds = engine.psycho_model.compute_masking_thresholds(in_power, sfm)
    
    nmr = 10.0 * torch.log10((err_power / (thresholds + 1e-12)) + 1e-12)
    return [float(v) for v in torch.max(nmr, dim=0).values.cpu().numpy()]


def compute_transient_smear_index(x_in: torch.Tensor, x_out: torch.Tensor, delay_samples: int = 0) -> float:
    if x_in.shape[-1] <= delay_samples + 2:
        return 1.0
    x_in_aligned = x_in[:, :-delay_samples] if delay_samples > 0 else x_in
    x_out_aligned = x_out[:, delay_samples:] if delay_samples > 0 else x_out
    
    grad_in = torch.diff(x_in_aligned, dim=-1)
    grad_out = torch.diff(x_out_aligned, dim=-1)
    norm_in = torch.sqrt(torch.sum(grad_in ** 2) + 1e-12)
    norm_out = torch.sqrt(torch.sum(grad_out ** 2) + 1e-12)
    dot = torch.sum(grad_in * grad_out)
    return float((dot / (norm_in * norm_out)).item())


def compute_stereo_correlation(x: torch.Tensor) -> float:
    if x.shape[0] < 2:
        return 1.0
    l, r = x[0], x[1]
    norm_l = torch.sqrt(torch.sum(l ** 2) + 1e-12)
    norm_r = torch.sqrt(torch.sum(r ** 2) + 1e-12)
    return float((torch.sum(l * r) / (norm_l * norm_r)).item())


def print_banner():
    print("=" * 86)
    print("       DYNAMIC PSYCHOACOUSTIC MULTI-BAND LIMITER // AUDIT HARNESS")
    print("             Mel/CELT-Scale Transient De-Grouping Pipeline")
    print("=" * 86)


def print_breakdown_table(report: dict):
    print("\n" + "-" * 86)
    print(f"  DIAGNOSTIC TELEMETRY BREAKDOWN : {report['filename']} [{report['mode'].upper()}]")
    print("-" * 86)
    print(f"  Input True-Peak     : {report['input_true_peak_dbtp']:+7.2f} dBTP  | Output True-Peak    : {report['output_true_peak_dbtp']:+7.2f} dBTP")
    print(f"  Input Peak (1x)     : {report['input_peak_dbfs']:+7.2f} dBFS  | Output Peak (1x)    : {report['output_peak_dbfs']:+7.2f} dBFS")
    print(f"  Max Gain Reduction  : {report['max_gain_reduction_db']:7.2f} dB    | Calibrated Delay    : {report['calibrated_delay_samples']:7d} samples")
    print(f"  Stereo Corr (In/Out): {report['stereo_correlation_in']:6.4f} / {report['stereo_correlation_out']:6.4f} | Phase Delta         : {report['stereo_phase_delta']:10.2e}")
    print(f"  Transient Smear TSI : {report['transient_smear_index']:7.4f}       | Max NMR Violation   : {max(report['celt_band_max_nmr_db']):+7.2f} dB")
    print(f"  Gain Accel (d2g/dt2): {report['gain_acceleration_c2_max']:10.2e}      | MDCT Intersample    : {report['inter_sample_clip_count']:7d}")
    print(f"  Ceiling Compliance  : {str(report['is_true_peak_compliant']):<5}       | Bit-Exact Passband  : {str(report['is_bit_exact_passband'])}")
    print("-" * 86 + "\n")


@torch.inference_mode()
def process_file_with_config(path: str, config: LimiterConfig, engine: PsychoacousticLimiterEngine, output_dir: str) -> dict:
    data, sr = sf.read(path, dtype="float32")
    tensor_in = torch.from_numpy(data if data.ndim > 1 else data[:, None]).T
    channels = tensor_in.shape[0]
    
    if config.prepass_enabled:
        tensor_out = engine.process_full_prepass(tensor_in).cpu()
    else:
        state = LimiterState()
        state.reset(channels=channels, config=config, device=engine.device)
        batch_block_size = 4800
        chunks = torch.split(tensor_in, batch_block_size, dim=-1)
        out_chunks = []
        for c in chunks:
            out_chunks.append(engine.process_block(c, state).cpu())
        tensor_out = torch.cat(out_chunks, dim=-1)

    base_name, ext = os.path.splitext(os.path.basename(path))
    mode_tag = "mel" if config.mode == "psychoacoustic_celt" else "fir"
    prepass_tag = "-prepass" if config.prepass_enabled else ""
    out_filename = f"{base_name}-{mode_tag}{prepass_tag}{ext}"
    
    out_path = os.path.join(output_dir, out_filename)
    sf.write(out_path, tensor_out.numpy().T, sr, subtype="FLOAT")

    in_tp = engine.resampler.measure_true_peak(tensor_in.to(engine.device))
    out_tp = engine.resampler.measure_true_peak(tensor_out.to(engine.device))
    
    in_peak_1x = float(20.0 * np.log10(max(np.max(np.abs(data)), 1e-12)))
    out_peak_1x = float(20.0 * np.log10(max(torch.max(torch.abs(tensor_out)).item(), 1e-12)))
    
    best_lag = find_system_delay_lag(tensor_in, tensor_out)
    
    is_sub_peak = in_tp <= config.true_peak_ceiling_db
    res_linf = 0.0
    if is_sub_peak and tensor_in.shape[-1] > best_lag:
        aligned_in = tensor_in[:, :-best_lag] if best_lag > 0 else tensor_in
        aligned_out = tensor_out[:, best_lag:] if best_lag > 0 else tensor_out
        res_linf = float(torch.max(torch.abs(aligned_in - aligned_out)).item())

    tsi = compute_transient_smear_index(tensor_in, tensor_out, delay_samples=best_lag)
    stereo_corr_in = compute_stereo_correlation(tensor_in)
    stereo_corr_out = compute_stereo_correlation(tensor_out)
    stereo_delta = abs(stereo_corr_in - stereo_corr_out)
    
    nmr_per_band = compute_empirical_nmr(tensor_in, tensor_out, engine, delay_samples=best_lag)
    clip_count = int(np.sum(tensor_out.numpy() > config.target_ceiling_linear + 1e-4))
    max_gr_db = float(max(0.0, in_tp - out_tp))
    
    envelope = torch.max(torch.abs(tensor_out), dim=0).values.numpy()
    gain_estimate = np.clip(config.target_ceiling_linear / (envelope + 1e-12), 0.0, 1.0)
    accel_max = float(np.max(np.abs(np.diff(gain_estimate, n=2))))

    report = AuditResult(
        filename=out_filename,
        mode=f"{config.mode}{'-prepass' if config.prepass_enabled else ''}",
        passband_residual_linf=res_linf,
        input_true_peak_dbtp=in_tp,
        output_true_peak_dbtp=out_tp,
        input_peak_dbfs=in_peak_1x,
        output_peak_dbfs=out_peak_1x,
        max_gain_reduction_db=max_gr_db,
        stereo_correlation_in=stereo_corr_in,
        stereo_correlation_out=stereo_corr_out,
        stereo_phase_delta=stereo_delta,
        transient_smear_index=tsi,
        calibrated_delay_samples=best_lag,
        celt_band_max_nmr_db=nmr_per_band,
        gain_acceleration_c2_max=accel_max,
        inter_sample_clip_count=clip_count,
        is_bit_exact_passband=bool(res_linf < 1e-5 if is_sub_peak else True),
        is_true_peak_compliant=bool(out_tp <= config.true_peak_ceiling_db + 0.005),
        is_psychoacoustically_transparent=bool(tsi >= 0.98 and max(nmr_per_band) <= 0.0)
    )
    
    del tensor_in, tensor_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return report.__dict__


def run_audit(input_dir: str, output_dir: str, configs_to_run: list):
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print(f"\n[!] No .wav files found in {input_dir}")
        return

    all_reports = []
    t_start = time.time()

    for config in configs_to_run:
        mode_desc = f"{config.mode.upper()}{' (PRE-PASS GLOBAL)' if config.prepass_enabled else ' (STREAMING LOOKAHEAD)'}"
        print(f"\n[*] Executing Pipeline Mode: {mode_desc}")
        print(f"    Target Ceiling : {config.true_peak_ceiling_db:.2f} dBTP | Lookahead: {config.lookahead_samples} samples | Oversampling: {config.oversample_factor}x")
        print(f"    CELT Subbands  : {config.num_celt_bands} bands | Stereo Link: {config.stereo_link} | Device: {config.device}")

        engine = PsychoacousticLimiterEngine(config)
        
        for path in wav_files:
            rep = process_file_with_config(path, config, engine, output_dir)
            all_reports.append(rep)
            print_breakdown_table(rep)

    t_elapsed = time.time() - t_start
    manifest_path = os.path.join(output_dir, "audit_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"[+] Audit completed in {t_elapsed:.2f}s | Telemetry written to {manifest_path}")
    print("=" * 86)


def interactive_cli_menu(default_config_path: str, default_input: str, default_output: str):
    print_banner()
    base_cfg = load_config(default_config_path)
    
    current_input = default_input
    current_output = default_output
    current_mode = "psychoacoustic_celt"
    current_ceiling = base_cfg.true_peak_ceiling_db
    current_lookahead = base_cfg.lookahead_samples
    current_oversample = base_cfg.oversample_factor
    current_stereo_link = base_cfg.stereo_link
    current_prepass = True
    current_device = base_cfg.device

    while True:
        print("\n" + "=" * 86)
        print("  HARNESS INTERACTIVE CONFIGURATION & AUDIT MENU")
        print("=" * 86)
        print(f"  [1] Execution Mode       : {current_mode.upper()}")
        print(f"  [2] Pre-Pass Engine      : {'ENABLED (Global Spline Optimization)' if current_prepass else 'DISABLED (Streaming Block Engine)'}")
        print(f"  [3] Target Ceiling       : {current_ceiling:.2f} dBTP")
        print(f"  [4] Lookahead Window     : {current_lookahead} samples ({current_lookahead / 48.0:.2f} ms)")
        print(f"  [5] Polyphase Oversample : {current_oversample}x")
        print(f"  [6] Stereo Coupling Link : {'ENABLED (Locked Phase)' if current_stereo_link else 'DISABLED (Dual Mono)'}")
        print(f"  [7] Compute Device       : {current_device.upper()}")
        print(f"  [8] Input Directory      : {current_input}")
        print(f"  [9] Output Directory     : {current_output}")
        print("-" * 86)
        print("  [R] Run Limiter Audit with Current Configuration")
        print("  [S] Run Full Permutation Benchmark Sweep (CELT [-mel] + FIR [-fir])")
        print("  [Q] Exit")
        print("=" * 86)

        choice = input("Select an option: ").strip().lower()

        if choice == "1":
            print("\nSelect Limiting Algorithm:")
            print("  1. FIR Linear-Phase Lookahead Limiting (Outputs *-fir*.wav)")
            print("  2. Psychoacoustic CELT Multiband Limiting (Outputs *-mel*.wav)")
            m_choice = input("Choice (1-2): ").strip()
            if m_choice == "1":
                current_mode = "fir_linear_phase"
            elif m_choice == "2":
                current_mode = "psychoacoustic_celt"

        elif choice == "2":
            current_prepass = not current_prepass
            print(f"[*] Pre-Pass Engine set to: {current_prepass}")

        elif choice == "3":
            val = input(f"Enter target true-peak ceiling in dBTP (current: {current_ceiling:.2f}): ").strip()
            try:
                current_ceiling = float(val)
            except ValueError:
                print("[!] Invalid numeric input.")

        elif choice == "4":
            print("\nSelect Lookahead Window Size:")
            print("  1. 96 samples (2.0 ms)")
            print("  2. 120 samples (2.5 ms)")
            print("  3. 144 samples (3.0 ms - Default)")
            print("  4. 192 samples (4.0 ms)")
            l_choice = input("Choice (1-4): ").strip()
            look_map = {"1": 96, "2": 120, "3": 144, "4": 192}
            if l_choice in look_map:
                current_lookahead = look_map[l_choice]

        elif choice == "5":
            print("\nSelect Polyphase Oversampling Factor:")
            print("  1. 4x (192 kHz)")
            print("  2. 8x (384 kHz - Broadcast Reference)")
            os_choice = input("Choice (1-2): ").strip()
            if os_choice == "1":
                current_oversample = 4
            elif os_choice == "2":
                current_oversample = 8

        elif choice == "6":
            current_stereo_link = not current_stereo_link
            print(f"[*] Stereo Phase Lock set to: {current_stereo_link}")

        elif choice == "7":
            current_device = "cpu" if current_device == "cuda" else "cuda"
            print(f"[*] Device set to: {current_device}")

        elif choice == "8":
            val = input(f"Enter input directory (current: {current_input}): ").strip()
            if val:
                current_input = val

        elif choice == "9":
            val = input(f"Enter output directory (current: {current_output}): ").strip()
            if val:
                current_output = val

        elif choice == "r":
            cfg = LimiterConfig(
                sample_rate=base_cfg.sample_rate,
                chunk_size=base_cfg.chunk_size,
                lookahead_samples=current_lookahead,
                oversample_factor=current_oversample,
                true_peak_ceiling_db=current_ceiling,
                mode=current_mode,
                num_celt_bands=base_cfg.num_celt_bands,
                smoothness_weight=base_cfg.smoothness_weight,
                transient_weight=base_cfg.transient_weight,
                spectral_flatness_offset_tone=base_cfg.spectral_flatness_offset_tone,
                spectral_flatness_offset_noise=base_cfg.spectral_flatness_offset_noise,
                basilar_compression_exponent=base_cfg.basilar_compression_exponent,
                ath_min_db=base_cfg.ath_min_db,
                stereo_link=current_stereo_link,
                prepass_enabled=current_prepass,
                device=current_device
            )
            run_audit(current_input, current_output, [cfg])

        elif choice == "s":
            cfg1 = LimiterConfig(
                sample_rate=base_cfg.sample_rate,
                chunk_size=base_cfg.chunk_size,
                lookahead_samples=current_lookahead,
                oversample_factor=current_oversample,
                true_peak_ceiling_db=current_ceiling,
                mode="fir_linear_phase",
                num_celt_bands=base_cfg.num_celt_bands,
                smoothness_weight=base_cfg.smoothness_weight,
                transient_weight=base_cfg.transient_weight,
                spectral_flatness_offset_tone=base_cfg.spectral_flatness_offset_tone,
                spectral_flatness_offset_noise=base_cfg.spectral_flatness_offset_noise,
                basilar_compression_exponent=base_cfg.basilar_compression_exponent,
                ath_min_db=base_cfg.ath_min_db,
                stereo_link=current_stereo_link,
                prepass_enabled=current_prepass,
                device=current_device
            )
            cfg2 = LimiterConfig(
                sample_rate=base_cfg.sample_rate,
                chunk_size=base_cfg.chunk_size,
                lookahead_samples=current_lookahead,
                oversample_factor=current_oversample,
                true_peak_ceiling_db=current_ceiling,
                mode="psychoacoustic_celt",
                num_celt_bands=base_cfg.num_celt_bands,
                smoothness_weight=base_cfg.smoothness_weight,
                transient_weight=base_cfg.transient_weight,
                spectral_flatness_offset_tone=base_cfg.spectral_flatness_offset_tone,
                spectral_flatness_offset_noise=base_cfg.spectral_flatness_offset_noise,
                basilar_compression_exponent=base_cfg.basilar_compression_exponent,
                ath_min_db=base_cfg.ath_min_db,
                stereo_link=current_stereo_link,
                prepass_enabled=current_prepass,
                device=current_device
            )
            run_audit(current_input, current_output, [cfg1, cfg2])

        elif choice == "q":
            print("[*] Exiting audit harness.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limiter Dual-Regime Audit Harness")
    parser.add_argument("--input_dir", type=str, default="workspace/input")
    parser.add_argument("--output_dir", type=str, default="workspace/output")
    parser.add_argument("--config", type=str, default="core/config/inference/Limiter_Opus_48k.yaml")
    parser.add_argument("--mode", type=str, choices=["psychoacoustic_celt", "fir_linear_phase"], default="psychoacoustic_celt")
    parser.add_argument("--prepass", action="store_true", default=True, help="Enable two-pass global spline optimization")
    parser.add_argument("--streaming", action="store_false", dest="prepass", help="Disable prepass and run streaming lookahead")
    parser.add_argument("--sweep", action="store_true", help="Run full permutation benchmark sweep (FIR + CELT)")
    parser.add_argument("--cli", action="store_true", help="Launch interactive CLI menu")
    args = parser.parse_args()

    if args.cli or (len(sys.argv) == 1):
        interactive_cli_menu(args.config, args.input_dir, args.output_dir)
    else:
        print_banner()
        if args.sweep:
            c1 = load_config(args.config)
            c1.mode = "fir_linear_phase"
            c1.prepass_enabled = args.prepass
            c2 = load_config(args.config)
            c2.mode = "psychoacoustic_celt"
            c2.prepass_enabled = args.prepass
            configs = [c1, c2]
        else:
            c = load_config(args.config)
            c.mode = args.mode
            c.prepass_enabled = args.prepass
            configs = [c]
        run_audit(args.input_dir, args.output_dir, configs)