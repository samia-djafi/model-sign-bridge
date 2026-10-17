"""
BridgeSign — Every Hand Has a Voice: Unified Multi-Mode Application Runner.

Provides an interactive console interface supporting:
  [1] Alphabet / Fingerspelling Mode (Algerian manual alphabet with buffer & editing)
  [2] Medical & Emergency Gesture Recognition Mode (Landmark classifier & Clinical Assembler)
  [3] Healthcare Sentence Library Mode (100 Practical Sentences with French/Arabic TTS)
  [4] Full System Inventory & Status Audit
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.alphabet.alphabet_inventory import ALGERIAN_ALPHABET
from src.alphabet.fingerspelling_buffer import FingerspellingBuffer
from src.data.vocabulary_inventory import TARGET_VOCABULARY_150, get_inventory_summary
from src.translation.clinical_assembler import ClinicalAssembler, SIGN_METADATA
from src.translation.sentence_library import HEALTHCARE_SENTENCE_LIBRARY, get_library_summary
from src.translation.tts import TextToSpeechSynthesizer


def print_banner():
    print("=" * 75)
    print("   BRIDGESIGN (كل يد لها صوت) — ALGERIAN SIGN LANGUAGE RECOGNITION SYSTEM   ")
    print("=" * 75)


def run_alphabet_mode(tts: TextToSpeechSynthesizer):
    buffer = FingerspellingBuffer(debounce_seconds=0.3)
    print("\n--- [MODE 1: ALPHABET & FINGERSPELLING] ---")
    print("Available letters: 28 Arabic letters + ة, لا, ء (31 total)")
    print("Commands:")
    print("  Type any Arabic letter to append.")
    print("  Type 'back' or '<' to delete the last character.")
    print("  Type 'space' or '_' to add a word separation.")
    print("  Type 'done' or 'fin' to finalize and pronounce the text.")
    print("  Type 'clear' to reset buffer. Type 'menu' to return.\n")

    while True:
        try:
            cmd = input(f"Buffer [{buffer.current_text}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue
        if cmd.lower() == "menu":
            break
        elif cmd.lower() == "clear":
            buffer.clear()
            print("Buffer cleared.")
        elif cmd.lower() in ["back", "<"]:
            char = buffer.backspace()
            print(f"Deleted: '{char}' | Current: \"{buffer.current_text}\"")
        elif cmd.lower() in ["space", "_"]:
            buffer.add_space()
            print(f"Added space | Current: \"{buffer.current_text}\"")
        elif cmd.lower() in ["done", "fin"]:
            final_text = buffer.finalize()
            translit = buffer.get_transliteration()
            print(f"\nFinalized Text: \"{final_text}\" (Transliteration: {translit})")
            if final_text:
                print(f"[TTS] Pronouncing: \"{final_text}\"...")
                tts.speak(final_text, lang="ar")
        else:
            # Check if letter is valid
            for ch in cmd:
                if ch == " ":
                    buffer.add_space()
                elif ch in ALGERIAN_ALPHABET:
                    info = ALGERIAN_ALPHABET[ch]
                    buffer.add_letter(ch)
                    print(f"  + Letter '{ch}' ({info['arabic_name']} / {info['french_name']})")
                else:
                    print(f"  ! Unknown symbol '{ch}' (Not in verified Algerian LSA alphabet)")
            print(f"  Current Buffer: \"{buffer.current_text}\"")


def run_gesture_mode(tts: TextToSpeechSynthesizer):
    assembler = ClinicalAssembler()
    print("\n--- [MODE 2: MEDICAL & EMERGENCY GESTURE RECOGNITION] ---")
    print("Enter Arabic sign glosses separated by spaces, or choose a preset:")
    print("  Presets: 'headache', 'emergency', 'posology', 'dental'")
    print("  Commands: 'assemble' to generate record, 'clear', 'menu'\n")

    presets = {
        "headache": ["من فضلك", "مريض", "ألم", "رأس", "صباح"],
        "emergency": ["خطر", "حرارة", "مريض", "الآن", "يساعد"],
        "posology": ["1", "نصف", "كل يوم", "ماء", "طعام"],
        "dental": ["ألم", "طبيب أسنان", "الليل", "من فضلك"],
    }

    while True:
        try:
            line = input("Signs > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line.lower() == "menu":
            break
        elif line.lower() == "clear":
            assembler.clear()
            print("Buffer reset.")
            continue
        elif line.lower() == "assemble":
            res = assembler.assemble()
            print("\n" + "=" * 65)
            print(f"Urgency Level: [{res.urgency_level}]")
            print(f"French: \"{res.french_sentence}\"")
            print(f"Arabic: \"{res.arabic_sentence}\"")
            print(f"English: \"{res.english_sentence}\"")
            print("=" * 65)
            print(f"[TTS] Speaking French sentence: \"{res.french_sentence}\"...")
            tts.speak(res.french_sentence, lang="fr")
            assembler.clear()
            continue

        tokens = presets.get(line.lower(), line.split())
        for tok in tokens:
            if tok in SIGN_METADATA:
                assembler.add_token(tok)
                print(f"  + Recognized: {tok} ({SIGN_METADATA[tok]['fr']})")
            else:
                print(f"  ! Sign '{tok}' not in current 32-sign active model")
        print(f"Current Tokens: {' -> '.join(assembler.tokens)}")


def run_sentence_library_mode(tts: TextToSpeechSynthesizer):
    print("\n--- [MODE 3: HEALTHCARE SENTENCE LIBRARY (100 SENTENCES)] ---")
    categories = sorted(list(set(s["category"] for s in HEALTHCARE_SENTENCE_LIBRARY)))
    print("Categories available:")
    for idx, cat in enumerate(categories, 1):
        count = sum(1 for s in HEALTHCARE_SENTENCE_LIBRARY if s["category"] == cat)
        print(f"  [{idx}] {cat} ({count} sentences)")
    print("  [8] Search by keyword")
    print("  [9] Return to Main Menu\n")

    while True:
        try:
            choice = input("Select category [1-9] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "9" or choice.lower() == "menu":
            break
        elif choice == "8":
            query = input("Enter search keyword (French, Arabic, or English) > ").strip().lower()
            matches = [
                s for s in HEALTHCARE_SENTENCE_LIBRARY
                if query in s["french_text"].lower() or query in s["arabic_text"] or query in s["english_gloss"].lower()
            ]
            print(f"\nFound {len(matches)} matching sentence(s):")
            for m in matches[:10]:
                print(f"  [{m['sentence_id']}] [{m['category']}] ({m['generation_mode']})")
                print(f"    FR: {m['french_text']}")
                print(f"    AR: {m['arabic_text']}")
            if matches:
                speak_id = input("\nEnter Sentence ID to speak aloud (or Enter to skip) > ").strip().upper()
                selected = next((s for s in matches if s["sentence_id"] == speak_id), None)
                if selected:
                    print(f"[TTS] Speaking: \"{selected['french_text']}\"...")
                    tts.speak(selected["french_text"], lang="fr")
        elif choice.isdigit() and 1 <= int(choice) <= len(categories):
            cat = categories[int(choice) - 1]
            cat_sentences = [s for s in HEALTHCARE_SENTENCE_LIBRARY if s["category"] == cat]
            print(f"\nSentences in '{cat}':")
            for idx, s in enumerate(cat_sentences, 1):
                print(f"  {idx}. [{s['sentence_id']}] {s['french_text']} ({s['arabic_text']})")
            sub = input(f"\nSelect sentence [1-{len(cat_sentences)}] to pronounce (or Enter to back) > ").strip()
            if sub.isdigit() and 1 <= int(sub) <= len(cat_sentences):
                target = cat_sentences[int(sub) - 1]
                print(f"[TTS] Speaking: \"{target['french_text']}\"...")
                tts.speak(target["french_text"], lang="fr")


def display_audit_summary():
    print("\n--- [SYSTEM INVENTORY & AUDIT SUMMARY] ---")
    vocab_summary = get_inventory_summary()
    sent_summary = get_library_summary()

    print(f"Total Target Vocabulary Concepts : {vocab_summary['total_target_concepts']}")
    print(f"  - Medical & Pharmacy (MED)     : {vocab_summary['category_breakdown']['Medical & Pharmacy']}")
    print(f"  - Emergency (EMG)              : {vocab_summary['category_breakdown']['Emergency']}")
    print(f"  - General Healthcare (GEN)     : {vocab_summary['category_breakdown']['General Healthcare']}")
    print("\nVocabulary Implementation Status:")
    for status, count in vocab_summary["status_counts"].items():
        print(f"  * {status:20s}: {count} concepts")

    print(f"\nTotal Healthcare Sentences       : {sent_summary['total_sentences']}")
    print("  - Category Distribution:")
    for cat, cnt in sent_summary["category_distribution"].items():
        print(f"    * {cat:30s}: {cnt}")
    print("  - Generation Modes:")
    for mode, cnt in sent_summary["generation_modes"].items():
        print(f"    * {mode:25s}: {cnt}")

    print(f"\nAlphabet Coverage               : {len(ALGERIAN_ALPHABET)} letters cataloged (0 trained in VDzSL)")
    print("-" * 75)


def main():
    parser = argparse.ArgumentParser(description="BridgeSign Unified Multi-Mode Runner")
    parser.add_argument("--mode", choices=["alphabet", "gesture", "sentences", "audit"], help="Launch directly into a specific mode")
    args = parser.parse_args()

    tts = TextToSpeechSynthesizer(enabled=True)

    if args.mode == "alphabet":
        run_alphabet_mode(tts)
        tts.stop()
        return
    elif args.mode == "gesture":
        run_gesture_mode(tts)
        tts.stop()
        return
    elif args.mode == "sentences":
        run_sentence_library_mode(tts)
        tts.stop()
        return
    elif args.mode == "audit":
        display_audit_summary()
        tts.stop()
        return

    print_banner()
    while True:
        print("\nSelect an operational mode:")
        print("  [1] Alphabet & Fingerspelling Mode (31 letters, interactive buffer, backspace, finalize)")
        print("  [2] Medical & Emergency Gesture Recognition Mode (Landmark classifier & Clinical record)")
        print("  [3] Healthcare Sentence Library Mode (100 sentences, French/Arabic TTS)")
        print("  [4] Full System Inventory & Status Audit")
        print("  [5] Exit Application")

        try:
            choice = input("\nEnter choice [1-5] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "1":
            run_alphabet_mode(tts)
        elif choice == "2":
            run_gesture_mode(tts)
        elif choice == "3":
            run_sentence_library_mode(tts)
        elif choice == "4":
            display_audit_summary()
        elif choice == "5" or choice.lower() in ["q", "quit", "exit"]:
            print("Shutting down BridgeSign. Goodbye!")
            break
        else:
            print("Invalid choice, please select 1-5.")

    tts.stop()


if __name__ == "__main__":
    main()
