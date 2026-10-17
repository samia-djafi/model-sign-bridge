"""
Text-to-Speech (TTS) integration for BridgeSign.

Provides non-blocking audio synthesis for recognized healthcare sentences,
supporting French and Arabic output with graceful offline/dummy fallbacks
when audio hardware or specific voices are unavailable.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VoiceInfo:
    """Metadata describing an installed TTS voice."""
    voice_id: str
    name: str
    languages: List[str]
    gender: Optional[str] = None


class TextToSpeechSynthesizer:
    """Thread-safe Text-to-Speech synthesizer supporting multi-language speech."""

    def __init__(self, rate: int = 150, volume: float = 1.0, enabled: bool = True):
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.enabled = enabled
        self._speech_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._available_voices: List[VoiceInfo] = []
        self._backend = "none"

        if self.enabled:
            self._init_backend()
            self._start_worker()

    def _init_backend(self) -> None:
        """Initializes the pyttsx3 engine and categorizes available voices."""
        try:
            import pyttsx3
            # Initialize engine once to query voices
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            raw_voices = engine.getProperty("voices")
            for v in raw_voices:
                langs = []
                if hasattr(v, "languages") and v.languages:
                    langs = [str(l).lower() for l in v.languages]
                else:
                    # Parse from name / id
                    lower_name = (v.name or "").lower()
                    if "french" in lower_name or "fr" in lower_name or "hortense" in lower_name or "julie" in lower_name or "paul" in lower_name:
                        langs.append("fr")
                    if "arabic" in lower_name or "ar" in lower_name or "naayf" in lower_name or "hoda" in lower_name:
                        langs.append("ar")
                    if "english" in lower_name or "en" in lower_name or "david" in lower_name or "zira" in lower_name:
                        langs.append("en")

                self._available_voices.append(
                    VoiceInfo(
                        voice_id=str(v.id),
                        name=str(v.name),
                        languages=langs,
                        gender=getattr(v, "gender", None),
                    )
                )

            # Cleanup initial engine instance
            del engine
            self._backend = "pyttsx3"
            logger.info("Initialized pyttsx3 TTS backend with %d voices", len(self._available_voices))
        except Exception as exc:
            logger.warning("TTS backend initialization failed (falling back to dummy mode): %s", exc)
            self._backend = "dummy"

    def _start_worker(self) -> None:
        """Starts worker thread for non-blocking speech requests."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background loop consuming speech requests."""
        engine = None
        if self._backend == "pyttsx3":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
            except Exception as exc:
                logger.error("Failed to initialize worker pyttsx3 engine: %s", exc)
                engine = None

        while not self._stop_event.is_set():
            try:
                item = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            text, lang = item
            try:
                if engine is not None:
                    # Select voice matching target language if found
                    target_voice_id = self._find_voice_for_lang(lang)
                    if target_voice_id:
                        engine.setProperty("voice", target_voice_id)

                    engine.say(text)
                    engine.runAndWait()
                else:
                    logger.debug("[Dummy TTS Speak (%s)]: %s", lang, text)
            except Exception as exc:
                logger.warning("Error during speech synthesis: %s", exc)
            finally:
                self._speech_queue.task_done()

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def _find_voice_for_lang(self, lang: str) -> Optional[str]:
        """Finds matching voice ID for requested ISO language code."""
        lang_lower = lang.lower()
        for v in self._available_voices:
            if any(lang_lower in l for l in v.languages):
                return v.voice_id
        # Fallback: check name substring
        for v in self._available_voices:
            if lang_lower == "fr" and ("french" in v.name.lower() or "fr" in v.name.lower()):
                return v.voice_id
            if lang_lower == "ar" and ("arabic" in v.name.lower() or "ar" in v.name.lower()):
                return v.voice_id
        return None

    def speak(self, text: str, lang: str = "fr", blocking: bool = False) -> bool:
        """
        Enqueues text to be spoken in the specified language.

        Args:
            text: Sentence or word to synthesize.
            lang: Language code ('fr', 'ar', 'en').
            blocking: If True, waits until speech completes before returning.

        Returns:
            bool: True if synthesis was dispatched or processed.
        """
        clean_text = text.strip()
        if not clean_text or not self.enabled:
            return False

        if blocking and self._backend == "pyttsx3":
            try:
                import pyttsx3
                eng = pyttsx3.init()
                eng.setProperty("rate", self.rate)
                eng.setProperty("volume", self.volume)
                target_v = self._find_voice_for_lang(lang)
                if target_v:
                    eng.setProperty("voice", target_v)
                eng.say(clean_text)
                eng.runAndWait()
                return True
            except Exception as exc:
                logger.warning("Blocking speak failed: %s", exc)
                return False

        self._speech_queue.put((clean_text, lang))
        if blocking:
            self._speech_queue.join()
        return True

    def stop(self) -> None:
        """Stops worker thread and speech processing."""
        self._stop_event.set()
        # Empty remaining queue items
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except Exception:
                break
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def is_available(self) -> bool:
        """Returns True if a hardware TTS engine is active."""
        return self._backend == "pyttsx3"

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Returns metadata of all recognized voices."""
        return [
            {
                "id": v.voice_id,
                "name": v.name,
                "languages": v.languages,
                "gender": v.gender,
            }
            for v in self._available_voices
        ]
