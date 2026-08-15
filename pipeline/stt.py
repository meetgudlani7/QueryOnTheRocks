"""
Speech-to-Text Module

Handles audio transcription using Groq Whisper.
"""

import time
import logging
from typing import Tuple, Optional
import httpx

from config import settings

logger = logging.getLogger(__name__)


class STTError(Exception):
    """Custom exception for STT errors."""
    pass


async def transcribe_audio(
    audio_data: bytes,
    format: str = "audio/wav",
    language: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Transcribe audio data using Groq Whisper.

    Args:
        audio_data: Raw audio bytes
        format: Audio format (e.g., "audio/wav", "audio/mp3")
        language: Language hint for transcription. Only pass this when the
            caller actually knows the spoken language — forcing a wrong
            hint (e.g. always "en") measurably degrades transcription of
            audio in any other language. Omit it to let Whisper's own
            language detection run, which is what the returned
            detected_language reflects.

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
            # No Content-Type here: httpx sets "multipart/form-data; boundary=..."
            # itself from `files=`. Setting it manually (as the previous version
            # did) omits the boundary and silently corrupts the request body.
        }

        # format may carry codec params (e.g. "audio/webm;codecs=opus") from
        # the browser's real MediaRecorder MIME type — strip them for the
        # filename extension, since Groq matches on extension and rejects
        # anything not in its fixed format list.
        extension = format.split("/")[-1].split(";")[0] or "wav"
        files = {
            "file": (f"audio.{extension}", audio_data, format),
        }

        # verbose_json is what makes Whisper's actually-detected language
        # available below — plain json only returns "text".
        data = {"model": model, "response_format": "verbose_json"}
        if language:
            data["language"] = language

        timeout_s = settings.STT_TIMEOUT_MS / 1000

        # One retry on transient failures only (timeout / connection issues /
        # 5xx) — a bad request (4xx: unsupported format, invalid key, etc.)
        # is retried is not going to succeed on a second try, so fail fast on those.
        last_error: Optional[Exception] = None
        for attempt in range(1, 3):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.post(url, headers=headers, files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    transcript = result.get("text", "")

                    # verbose_json returns Whisper's own detected language
                    # rather than blindly echoing back an input hint.
                    detected_language = result.get("language") or language or "unknown"

                    latency_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(f"STT completed in {latency_ms:.2f}ms (attempt {attempt})")

                    return transcript, detected_language

                if response.status_code < 500 or attempt == 2:
                    logger.error(f"Groq STT error: {response.status_code} - {response.text}")
                    raise STTError(f"Groq STT failed: {response.status_code} - {response.text}")

                logger.warning(f"Groq STT returned {response.status_code}, retrying once...")

            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = e
                if attempt == 2:
                    raise STTError(f"HTTP error: {e}") from e
                logger.warning(f"STT attempt {attempt} failed ({e}), retrying once...")

        raise STTError(f"Groq STT failed after retry: {last_error}")
            
    except STTError:
        raise  # already a clean, specific message from the retry loop above
    except Exception as e:
        logger.error(f"STT failed: {e}", exc_info=True)
        raise STTError(f"STT failed: {e}")
