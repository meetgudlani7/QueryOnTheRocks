"""
Speech-to-Text Module

Handles audio transcription using Groq Whisper.
"""

import time
import logging
from typing import Tuple, Optional
import httpx
import os

from config import settings

logger = logging.getLogger(__name__)


class STTError(Exception):
    """Custom exception for STT errors."""
    pass


async def transcribe_audio(
    audio_data: bytes,
    format: str = "audio/wav",
    language: str = "en",
) -> Tuple[str, str]:
    """
    Transcribe audio data using Groq Whisper.
    
    Args:
        audio_data: Raw audio bytes
        format: Audio format (e.g., "audio/wav", "audio/mp3")
        language: Language hint for transcription
        
    Returns:
        Tuple of (transcript, detected_language)
        
    Raises:
        STTError: If transcription fails
    """
    start_time = time.perf_counter()
    
    try:
        # Get Groq API settings
        api_key = settings.GROQ_API_KEY
        model = settings.GROQ_STT_MODEL
        
        if not api_key:
            raise STTError("GROQ_API_KEY not configured")
        
        # Prepare request
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "multipart/form-data",
        }
        
        files = {
            "file": ("audio." + format.split("/")[-1], audio_data, format),
        }
        
        data = {
            "model": model,
            "language": language,
        }
        
        # Send request to Groq
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            
            if response.status_code != 200:
                logger.error(f"Groq STT error: {response.status_code} - {response.text}")
                raise STTError(f"Groq STT failed: {response.status_code} - {response.text}")
            
            result = response.json()
            transcript = result.get("text", "")
            
            # Note: Groq Whisper may not always return language
            # For now, return the provided language hint
            detected_language = language
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"STT completed in {latency_ms:.2f}ms")
            
            return transcript, detected_language
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error during STT: {e}", exc_info=True)
        raise STTError(f"HTTP error: {e}")
    except Exception as e:
        logger.error(f"STT failed: {e}", exc_info=True)
        raise STTError(f"STT failed: {e}")
