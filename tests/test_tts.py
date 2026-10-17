"""
Unit tests for Text-to-Speech (TTS) synthesizer and fallback handling.
"""

import pytest
from src.translation.tts import TextToSpeechSynthesizer


def test_tts_initialization():
    """Verify synthesizer initializes without throwing exceptions."""
    tts = TextToSpeechSynthesizer(enabled=True)
    assert tts._backend in ["pyttsx3", "dummy", "none"]
    voices = tts.get_available_voices()
    assert isinstance(voices, list)
    tts.stop()


def test_tts_speak_enqueuing():
    """Verify speak requests are accepted and do not raise errors."""
    tts = TextToSpeechSynthesizer(enabled=True)
    res_fr = tts.speak("Bonjour, j'ai mal à la tête.", lang="fr", blocking=False)
    assert res_fr is True

    res_ar = tts.speak("مرحباً لدي ألم في الرأس", lang="ar", blocking=False)
    assert res_ar is True

    # Empty text should return False
    assert tts.speak("", lang="fr") is False
    tts.stop()


def test_tts_disabled_mode():
    """Verify synthesizer behavior when disabled."""
    tts = TextToSpeechSynthesizer(enabled=False)
    assert tts.speak("Test", lang="fr") is False
    tts.stop()
