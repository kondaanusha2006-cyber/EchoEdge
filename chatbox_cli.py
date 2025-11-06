# chatbox_cli.py
"""
Simple CLI Chatbox:
- Records a short audio clip from mic
- Verifies the speaker against owner_embedding.npy (Resemblyzer)
- If owner verified, transcribes using Whisper and prints transcript
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.spatial.distance import cosine

# Configuration
OWNER_EMBED_PATH = "owner_embedding.npy"
SAMPLE_RATE = 16000
RECORD_SECONDS = 3.0      # seconds to record
SIM_THRESHOLD = 0.65      # similarity threshold
WHISPER_MODEL = "tiny"    # change to "base" or "small" if you have CPU/GPU

# Helper: load owner embedding
def load_owner_embedding(path=OWNER_EMBED_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Owner embedding not found at {path}. Run enrollment first.")
    return np.load(path)

# Record to temporary file
def record_temp(path="temp_chat.wav", seconds=RECORD_SECONDS):
    print(f"🎙 Recording for {seconds:.1f} seconds. Speak now...")
    data = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, data, SAMPLE_RATE)
    print("✅ Saved:", path)
    return path

# Compute similarity using resemblyzer
def compute_similarity(file_path, owner_emb):
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder()
    wav = preprocess_wav(file_path)
    emb = encoder.embed_utterance(wav)
    sim = 1.0 - cosine(emb, owner_emb)
    return float(sim)

# Transcribe using Whisper
def transcribe(file_path):
    import whisper
    model = whisper.load_model(WHISPER_MODEL)
    res = model.transcribe(file_path)
    return res.get("text", "").strip()

# Main function
def main():
    try:
        owner_emb = load_owner_embedding()
    except Exception as e:
        print("❌ ERROR:", e)
        sys.exit(1)

    while True:
        choice = input("\nPress Enter to record a chat clip (or type 'q' to quit): ").strip().lower()
        if choice == "q":
            break

        tmp = record_temp()
        print("🔍 Verifying speaker...")
        try:
            sim = compute_similarity(tmp, owner_emb)
        except Exception as e:
            print("Verification error:", e)
            continue

        print(f"Similarity = {sim:.3f}")
        if sim < SIM_THRESHOLD:
            print("🚫 Access denied: speaker not recognized as owner.")
            continue

        print("✅ Owner verified. Transcribing...")
        try:
            text = transcribe(tmp)
            if text:
                print("\n--- Transcript ---")
                print(text)
                print("------------------\n")
            else:
                print("⚠ Transcription empty.")
        except Exception as e:
            print("Transcription error:", e)

# ✅ Correct entry point
if __name__ == "__main__":
    main()