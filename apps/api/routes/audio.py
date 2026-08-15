"""
Audio Processing Endpoint

Handles audio file upload and speech-to-text processing.
"""

from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import logging

from pipeline.orchestrator import process_audio
from pipeline.schemas import AudioRequest, AudioResponse

router = APIRouter(prefix="/api/audio", tags=["audio"])
logger = logging.getLogger(__name__)


@router.post("")
async def process_audio_file(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """
    Process uploaded audio file.

    Args:
        file: Audio file (WAV, MP3, etc.)
        language: Optional language hint (e.g. "en", "hi"). Omit to let
            Whisper auto-detect — recommended unless the caller actually
            knows the spoken language.

    Returns:
        AudioResponse: Transcript and processing metadata

    Raises:
        HTTPException: If file processing fails
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Expected audio file.")

        # Read file content
        audio_data = await file.read()

        # Process audio through pipeline
        request = AudioRequest(audio_data=audio_data, format=file.content_type, language=language)
        response = await process_audio(request)
        
        # mode="json" so datetime/etc. fields serialize to JSON-safe values
        # instead of raw Python objects (json.dumps can't encode a datetime).
        return JSONResponse(content=response.model_dump(mode="json"))

    except HTTPException:
        raise  # e.g. the 400 above — don't let the generic handler turn it into a 500
    except Exception as e:
        logger.error(f"Audio processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
