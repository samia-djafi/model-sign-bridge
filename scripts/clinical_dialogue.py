"""Clinical Dialogue CLI: Demonstrates converting recognized LSA signs into structured sentences and pharmacy triage records.

Usage:
  # Test with preset scenarios:
  python scripts/clinical_dialogue.py --scenario headache
  python scripts/clinical_dialogue.py --scenario emergency
  python scripts/clinical_dialogue.py --scenario posology

  # Test with custom signs:
  python scripts/clinical_dialogue.py --signs "من فضلك" "مريض" "ألم" "رأس" "صباح"

  # Interactive mode:
  python scripts/clinical_dialogue.py --interactive
"""

from __future__ import annotations
import argparse
import json
import sys
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.translation.clinical_assembler import ClinicalAssembler, SIGN_METADATA


PRESET_SCENARIOS = {
    "headache": [
        "من فضلك",
        "مريض",
        "ألم",
        "رأس",
        "صباح",
        "شكرا",
    ],
    "emergency": [
        "خطر",
        "حرارة",
        "مريض",
        "الآن",
        "يساعد",
    ],
    "posology": [
        "1",
        "نصف",
        "كل يوم",
        "ماء",
        "طعام",
        "كم",
        "مال",
    ],
    "dental": [
        "ألم",
        "طبيب أسنان",
        "الليل",
        "من فضلك",
        "يساعد",
    ],
    "diabetes": [
        "سكر",
        "مريض",
        "إبرة",
        "كحول",
        "كل يوم",
    ],
}


def print_banner():
    print("=" * 70)
    print("   BRIDGE SIGN - LSA CLINICAL TRANSLATION & PHARMACY ASSEMBLER   ")
    print("=" * 70)


def display_result(result):
    print("\n" + "-" * 70)
    print("  INPUT SIGNS (Glosses):")
    tokens_str = " -> ".join([f"[{t}]" for t in result.raw_tokens])
    print(f"  {tokens_str}")
    print("-" * 70)

    print("\n  [FRANÇAIS - PHRASE FLUIDE]:")
    print(f"  \"{result.french_sentence}\"")

    print("\n  [العربية - جملة واضحة ومترابطة]:")
    print(f"  \"{result.arabic_sentence}\"")

    print("\n  [ENGLISH - CLINICAL PHRASE]:")
    print(f"  \"{result.english_sentence}\"")

    print("\n" + "-" * 70)
    print(f"  FICHE COMPTOIR PHARMACIE (Triage & Dossier Médical) [Niveau: {result.urgency_level}]:")
    summary = result.clinical_summary
    print(f"    - État Patient         : {summary.get('patient_state', 'N/A')}")
    print(f"    - Symptôme(s)          : {', '.join(summary.get('symptomes', [])) or 'Aucun'}")
    print(f"    - Siège anatomique     : {', '.join(summary.get('siege_anatomique', [])) or 'Non spécifié'}")
    print(f"    - Temporalité          : {', '.join(summary.get('temporalite', [])) or 'Non spécifiée'}")
    
    poso = summary.get("posologie", {})
    if poso.get("quantite"):
        print(f"    - Posologie            : {poso.get('quantite')} unité(s) - Fréquence: {poso.get('frequence') or 'Ponctuelle'}")
        print(f"    - Modalité prise       : Eau={poso.get('avec_eau')} | Repas={poso.get('avec_repas')}")
    if summary.get("pathologies_connexes"):
        print(f"    - Terrain / Pathologie : {', '.join(summary.get('pathologies_connexes'))}")
    if summary.get("specialiste"):
        print(f"    - Spécialiste requis   : {', '.join(summary.get('specialiste'))}")
    print("-" * 70 + "\n")


def run_interactive():
    assembler = ClinicalAssembler(debounce_window_sec=1.0)
    print_banner()
    print("Mode interactif: Entrez des signes un par un ou séparés par des espaces.")
    print("Signes disponibles:", ", ".join(list(SIGN_METADATA.keys())[:15]), "...")
    print("Tapez 'fin' pour assembler, 'reset' pour effacer, 'quit' pour quitter.\n")

    while True:
        try:
            line = input("Signe(s) > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line.lower() == "quit":
            break
        if line.lower() == "reset":
            assembler.clear()
            print("Buffer effacé.")
            continue
        if line.lower() == "fin":
            res = assembler.assemble()
            display_result(res)
            assembler.clear()
            continue

        parts = line.split()
        for p in parts:
            if p in SIGN_METADATA:
                added = assembler.add_token(p)
                if added:
                    print(f"  + Ajouté: {p} ({SIGN_METADATA[p]['fr']})")
                else:
                    print(f"  ~ Doublon ignoré: {p}")
            else:
                print(f"  ! Signe non reconnu dans le vocabulaire MVP 32: {p}")


def main():
    parser = argparse.ArgumentParser(description="LSA Clinical Dialogue & Pharmacy Assembler CLI")
    parser.add_argument("--scenario", choices=list(PRESET_SCENARIOS.keys()), help="Run a predefined clinical scenario")
    parser.add_argument("--signs", nargs="+", help="List of Arabic sign labels to translate")
    parser.add_argument("--interactive", action="store_true", help="Start interactive terminal mode")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    parser.add_argument("--tts", action="store_true", help="Speak the translated sentence via Text-to-Speech")
    parser.add_argument("--tts-lang", choices=["fr", "ar"], default="fr", help="Language for TTS output (default: fr)")

    args = parser.parse_args()

    if args.interactive:
        run_interactive()
        return

    assembler = ClinicalAssembler()
    if args.scenario:
        signs = PRESET_SCENARIOS[args.scenario]
    elif args.signs:
        signs = args.signs
    else:
        print_banner()
        print("Aucun argument spécifié. Démonstration automatique du scénario 'headache' :\n")
        signs = PRESET_SCENARIOS["headache"]

    for s in signs:
        assembler.add_token(s)

    result = assembler.assemble()

    if args.json:
        payload = {
            "raw_tokens": result.raw_tokens,
            "french_sentence": result.french_sentence,
            "arabic_sentence": result.arabic_sentence,
            "english_sentence": result.english_sentence,
            "urgency_level": result.urgency_level,
            "clinical_summary": result.clinical_summary,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_banner()
        display_result(result)

    if args.tts:
        from src.translation.tts import TextToSpeechSynthesizer
        synth = TextToSpeechSynthesizer(enabled=True)
        spoken_text = result.french_sentence if args.tts_lang == "fr" else result.arabic_sentence
        print(f"\n[TTS ({args.tts_lang.upper()})]: Prononciation audio de: \"{spoken_text}\"...")
        synth.speak(spoken_text, lang=args.tts_lang, blocking=True)
        synth.stop()


if __name__ == "__main__":
    main()
