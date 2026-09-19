"""
Voice Assistant Module.
Provides Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities.
Designed modularly so the app functions seamlessly even if audio is unavailable.
"""
from __future__ import annotations
import io
import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Track library availability
_stt_available: bool = False
_tts_available: bool = False

try:
    import speech_recognition as sr
    _stt_available = True
except ImportError:
    logger.warning("SpeechRecognition not installed; STT will be disabled.")

try:
    from gtts import gTTS
    _tts_available = True
except ImportError:
    logger.warning("gTTS not installed; TTS will be disabled.")


def is_stt_available() -> bool:
    """Check if speech recognition is available."""
    return _stt_available


def is_tts_available() -> bool:
    """Check if text-to-speech is available."""
    return _tts_available


def transcribe_audio(audio_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio bytes (e.g. from st.audio_input) into text.

    Returns:
        (transcribed_text, error_message)
    """
    if not _stt_available:
        return None, "Speech recognition library is not available in the environment."

    if not audio_bytes or len(audio_bytes) < 100:
        return None, "No audio recorded."

    try:
        recognizer = sr.Recognizer()
        # Create audio file stream
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            # Adjust for ambient noise and record
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)

        # Transcribe using Google Web Speech API
        text = recognizer.recognize_google(audio_data)
        return text, None
    except sr.UnknownValueError:
        return None, "Could not understand the audio. Please try speaking closer to the microphone."
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        return None, f"Transcription error: {str(e)}"


def text_to_speech_bytes(text: str) -> Optional[bytes]:
    """
    Convert text response into MP3 audio bytes for browser playback.

    Returns:
        bytes of the MP3 file or None if TTS fails.
    """
    if not _tts_available or not text.strip():
        return None

    try:
        # Strip markdown symbols and citations for cleaner speech synthesis
        import re
        clean_text = re.sub(r"\[.*?\]", "", text)
        clean_text = re.sub(r"[*#_>`-]", "", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        # Limit to first 400 characters to ensure responsive fast audio generation
        if len(clean_text) > 400:
            clean_text = clean_text[:400] + "..."

        if not clean_text:
            return None

        fp = io.BytesIO()
        tts = gTTS(text=clean_text, lang="en", slow=False)
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return None
