"""
Interactive Fingerspelling Buffer and State Machine for Algerian Sign Language.

Allows patients and pharmacists to:
- Spell words letter-by-letter (patient names, specific drug trade names).
- Debounce consecutive letter additions.
- Delete/correct previous letters (backspace).
- Insert word separators (space).
- Finalize words and phrases.
- Produce dual Arabic script and French phonetic transliteration.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
from src.alphabet.alphabet_inventory import ALGERIAN_ALPHABET, get_alphabet_letter


# Transliteration mapping for Arabic alphabet to Latin/French phonetics
ARABIC_TO_FRENCH_TRANSLIT = {
    "أ": "A", "ب": "B", "ت": "T", "ث": "Th", "ج": "J", "ح": "H", "خ": "Kh",
    "د": "D", "ذ": "Dh", "ر": "R", "ز": "Z", "س": "S", "ش": "Ch", "ص": "S",
    "ض": "D", "ط": "T", "ظ": "Dh", "ع": "3", "غ": "Gh", "ف": "F", "ق": "Q",
    "ك": "K", "ل": "L", "م": "M", "ن": "N", "هـ": "H", "و": "W", "ي": "Y",
    "ة": "a", "لا": "La", "ء": "'", " ": " ",
}


@dataclass
class FingerspellingToken:
    """Represents a single fingerspelled letter event."""
    char: str
    confidence: float = 1.0
    timestamp: float = 0.0
    letter_id: str = ""
    arabic_name: str = ""
    french_name: str = ""

    def __post_init__(self):
        meta = get_alphabet_letter(self.char)
        if meta:
            self.letter_id = meta.get("letter_id", "")
            self.arabic_name = meta.get("arabic_name", "")
            self.french_name = meta.get("french_name", "")


class FingerspellingBuffer:
    """
    Manages interactive spelling sessions, letter-by-letter aggregation,
    temporal debouncing, backspace deletion, and word finalization.
    """

    def __init__(
        self,
        debounce_seconds: float = 1.2,
        min_confidence: float = 0.40,
    ):
        self.debounce_seconds = debounce_seconds
        self.min_confidence = min_confidence

        self.current_word_chars: List[FingerspellingToken] = []
        self.finalized_words: List[str] = []
        self.history: List[Dict[str, Any]] = []

        self.last_char: Optional[str] = None
        self.last_timestamp: float = 0.0

    def add_letter(
        self,
        char: str,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Add a letter to the current spelling buffer if confidence is sufficient
        and debounce interval has elapsed.
        """
        now = time.time() if timestamp is None else timestamp

        # Reject unknown characters not in alphabet
        if char not in ALGERIAN_ALPHABET and char != " ":
            return False

        # Reject low-confidence predictions
        if confidence < self.min_confidence:
            return False

        # Debounce identical consecutive letters
        if char == self.last_char:
            if (now - self.last_timestamp) < self.debounce_seconds:
                return False

        token = FingerspellingToken(
            char=char,
            confidence=confidence,
            timestamp=now,
        )

        if char == " ":
            self.finalize_word()
        else:
            self.current_word_chars.append(token)

        self.last_char = char
        self.last_timestamp = now
        self.history.append({
            "action": "add",
            "char": char,
            "confidence": confidence,
            "timestamp": now,
        })
        return True

    def backspace(self) -> bool:
        """Delete the most recently added letter."""
        if self.current_word_chars:
            removed = self.current_word_chars.pop()
            self.last_char = self.current_word_chars[-1].char if self.current_word_chars else None
            self.last_timestamp = time.time()
            self.history.append({"action": "backspace", "removed": removed.char})
            return True
        elif self.finalized_words:
            # Reopen the last finalized word
            reopened = self.finalized_words.pop()
            for ch in reopened:
                self.current_word_chars.append(FingerspellingToken(char=ch))
            self.last_char = self.current_word_chars[-1].char if self.current_word_chars else None
            return True
        return False

    def space(self) -> None:
        """Insert a word break and finalize the current word."""
        self.finalize_word()

    def clear(self) -> None:
        """Clear all active and finalized spelled words."""
        self.current_word_chars.clear()
        self.finalized_words.clear()
        self.last_char = None
        self.last_timestamp = 0.0
        self.history.append({"action": "clear"})

    def finalize_word(self) -> str:
        """Commit the current word buffer to the finalized word list."""
        if not self.current_word_chars:
            return ""
        word = "".join([t.char for t in self.current_word_chars])
        self.finalized_words.append(word)
        self.current_word_chars.clear()
        self.last_char = None
        self.history.append({"action": "finalize_word", "word": word})
        return word

    def get_current_word(self) -> str:
        """Get the current word being spelled."""
        return "".join([t.char for t in self.current_word_chars])

    def get_full_text(self) -> str:
        """Get complete assembled text including finalized words and current word."""
        words = list(self.finalized_words)
        curr = self.get_current_word()
        if curr:
            words.append(curr)
        return " ".join(words)

    def get_french_transliteration(self) -> str:
        """Convert current full text into French phonetic transliteration."""
        full_ar = self.get_full_text()
        parts = []
        for ch in full_ar:
            parts.append(ARABIC_TO_FRENCH_TRANSLIT.get(ch, ch))
        return "".join(parts)

    @property
    def current_text(self) -> str:
        """Alias property for get_full_text()."""
        return self.get_full_text()

    def add_space(self) -> None:
        """Alias for space()."""
        self.space()

    def finalize(self) -> str:
        """Finalize the current word and return the full text."""
        self.finalize_word()
        res = self.get_full_text()
        self.clear()
        return res

    get_transliteration = get_french_transliteration

    def get_summary(self) -> Dict[str, Any]:
        """Produce a machine-readable summary of the spelling state."""
        return {
            "current_word": self.get_current_word(),
            "finalized_words": list(self.finalized_words),
            "full_arabic_text": self.get_full_text(),
            "french_transliteration": self.get_french_transliteration(),
            "token_count": sum(len(w) for w in self.finalized_words) + len(self.current_word_chars),
            "is_empty": (len(self.current_word_chars) == 0 and len(self.finalized_words) == 0),
        }
