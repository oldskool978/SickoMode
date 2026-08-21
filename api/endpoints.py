import base64
import torch
import numpy as np
from fastapi import APIRouter, HTTPException
from api.schemas import AudioProcessRequest, AudioProcessResponse, HealthResponse
from core.schema import LimiterConfig, LimiterState
from core.engine import PsychoacousticLimiterEngine

router = APIRouter()
_CONFIG = LimiterConfig()
_ENGINE = PsychoacousticLimiterEngine(_CONFIG)
_STATE = LimiterState()
_STATE.reset(channels=2, config=_CONFIG, device=_ENGINE.device)


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        device=str(_ENGINE.device),
        oversample_factor=_CONFIG.oversample_factor,
        true_peak_ceiling_db=_CONFIG.true_peak_ceiling_db
    )


@router.post("/process-block", response_model=AudioProcessResponse)
async def process_block(request: AudioProcessRequest):
    try:
        raw_bytes = base64.b64decode(request.pcm_base64)
        audio_np = np.frombuffer(raw_bytes, dtype=np.float32)
        audio_np = audio_np.reshape((request.channels, -1))
        audio_tensor = torch.from_numpy(audio_np.copy())

        in_peak = _ENGINE.resampler.measure_true_peak(audio_tensor.to(_ENGINE.device))
        out_tensor = _ENGINE.process_block(audio_tensor, _STATE)
        out_peak = _ENGINE.resampler.measure_true_peak(out_tensor)

        out_np = out_tensor.detach().cpu().numpy().astype(np.float32)
        out_b64 = base64.b64encode(out_np.tobytes()).decode("utf-8")

        residual = 0.0
        if in_peak <= _CONFIG.true_peak_ceiling_db:
            residual = float(torch.max(torch.abs(audio_tensor.to(_ENGINE.device) - out_tensor)).item())

        return AudioProcessResponse(
            pcm_base64=out_b64,
            input_true_peak_dbtp=in_peak,
            output_true_peak_dbtp=out_peak,
            passband_residual_linf=residual,
            is_true_peak_compliant=bool(out_peak <= _CONFIG.true_peak_ceiling_db + 0.005)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))