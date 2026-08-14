"""
Audio Processing Endpoint

Handles audio file upload and speech-to-text processing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging

from pipeline.orchestrator import process_audio
from pipeline.schemas import AudioRequest, AudioResponse

router = APIRouter(prefix="/api/audio", tags=["audio"])
logger = logging.getLogger(__name__)


@router.post("")
async def process_audio_file(file: UploadFile = File(...)):
    """
    Process uploaded audio file.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
        
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
        request = AudioRequest(audio_data=audio_data, format=file.content_type)
        response = await process_audio(request)
        
        return JSONResponse(content=response.model_dump())
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
