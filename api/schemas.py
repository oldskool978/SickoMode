from pydantic import BaseModel
from typing import Optional


class AudioProcessRequest(BaseModel):
    session_id: Optional[str] = None
    pcm_base64: str
    channels: int = 2
    sample_rate: int = 48000


class AudioProcessResponse(BaseModel):
    session_id: str
    pcm_base64: str
    input_true_peak_dbtp: float
    output_true_peak_dbtp: float
    passband_residual_linf: float
    is_true_peak_compliant: bool


class HealthResponse(BaseModel):
    status: str
    device: str
    oversample_factor: int
    true_peak_ceiling_db: float