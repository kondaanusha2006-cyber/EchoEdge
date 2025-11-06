import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torchaudio

SAMPLERATE = 16000
DURATION = 4  # seconds
EMBED_PATH = "owner_embedding.npy"

model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()

def record_owner(filename):
    print("🎙️ Recording owner voice... Speak clearly now.")
    audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, SAMPLERATE)
    print("✅ Saved:", filename)

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
    record_owner("owner.wav")
    emb = get_embedding("owner.wav")
    np.save(EMBED_PATH, emb)
    print("✅ Owner embedding saved at:", EMBED_PATH)
