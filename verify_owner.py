import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torchaudio
import os  # to open calculator
from scipy.spatial.distance import cosine

# Settings
SAMPLERATE = 16000
DURATION = 3  # seconds for verification
EMBED_PATH = "owner_embedding.npy"
THRESHOLD = 0.75  # Adjust if needed

# Load speaker model
model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()

def record_voice(filename):
    print("🎙️ Speak to verify...")
    audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, SAMPLERATE)

def get_embedding(filename):
    wav, sr = torchaudio.load(filename)
    if sr != SAMPLERATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLERATE)

    with torch.no_grad():
        feats = model.extract_features(wav)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if isinstance(feats, list):
            feats = feats[0]
        emb = feats.mean(dim=1).numpy()
    return emb

if __name__ == "__main__":
    print("📁 Loading saved owner voice embedding...")
    owner = np.load(EMBED_PATH)

    record_voice("test.wav")
    test = get_embedding("test.wav")

    sim = 1 - cosine(owner.flatten(), test.flatten())
    print(f"\n🔍 Voice Similarity: {sim:.3f}")

    if sim >= THRESHOLD:
        print("✅ Access Granted — Owner Verified")
        print("🟢 Opening Calculator...")
        os.system("calc")  # ✅ Opens Windows Calculator
    else:
        print("❌ Access Denied — Not Owner")
