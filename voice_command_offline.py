# voice_command_offline.py
# Owner: EchoEdge — verify owner voice, transcribe command offline, open apps.
import os
import time
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torchaudio
import whisper
import webbrowser
from scipy.spatial.distance import cosine

# ---- Settings (tweak if needed) ----
SAMPLERATE = 16000
VERIFY_SECONDS = 3          # seconds for owner verification
COMMAND_SECONDS = 4         # seconds to capture spoken command
EMBED_PATH = "owner_embedding.npy"
SIMILARITY_THRESHOLD = 0.75 # strict owner threshold
ENERGY_THRESHOLD = 0.0001   # voice energy threshold (lower = more sensitive)
WHISPER_MODEL = "small"     # "tiny","base","small" -> accuracy vs speed

# ---- Map words to actions (customize paths) ----
# Keys are substrings to look for in transcribed command.
# Values: either a program name (os/system), full exe path, or a callable (lambda)
APP_MAP = {
    "calculator": {"type": "system", "cmd": "calc"},
    "notepad":     {"type": "system", "cmd": "notepad"},
    "paint":       {"type": "system", "cmd": "mspaint"},
    "explorer":    {"type": "system", "cmd": "explorer"},  # file explorer
    "chrome":      {"type": "path",   "cmd": r"C:\Program Files\Google\Chrome\Application\chrome.exe"},
    "vlc":         {"type": "path",   "cmd": r"C:\Program Files\VideoLAN\VLC\vlc.exe"},
    "whatsapp":    {"type": "path",   "cmd": r"C:\Users\prava\AppData\Local\WhatsApp\WhatsApp.exe"},
    "youtube":     {"type": "url",    "cmd": "https://www.youtube.com/"},
    # add your own mappings below
}

# ---- Load models lazily ----
_wav2vec_model = None
_whisper_model = None

def get_wav2vec():
    global _wav2vec_model
    if _wav2vec_model is None:
        _wav2vec_model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
    return _wav2vec_model

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model

# ---- Audio helpers ----
def record_to_file(path, duration_seconds):
    """Record `duration_seconds` seconds and write to WAV (mono, 16k)."""
    data = sd.rec(int(duration_seconds * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='float32')
    sd.wait()
    sf.write(path, data, SAMPLERATE)
    return path

def compute_energy(wav_path):
    data, sr = sf.read(wav_path)
    # If stereo, take first channel
    if data.ndim > 1:
        data = data[:,0]
    return float(np.mean(data.astype(np.float64) ** 2))

def get_embedding_from_file(path):
    """Return a numpy embedding using Wav2Vec2 pipeline (robust to torchaudio versions)."""
    model = get_wav2vec()
    wav, sr = torchaudio.load(path)
    if sr != SAMPLERATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLERATE)
    with torch.no_grad():
        feats = model.extract_features(wav)
        # handle list/tuple/tensor outputs
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if isinstance(feats, list):
            feats = feats[0]
        if not isinstance(feats, torch.Tensor):
            raise TypeError(f"Unexpected features type: {type(feats)}")
        emb = feats.mean(dim=1).cpu().numpy()
    return emb

def verify_owner(verify_wav="verify.wav"):
    """Record a short clip, check energy and similarity against stored owner embedding."""
    if not os.path.exists(EMBED_PATH):
        print("Owner embedding not found. Run enrollment first (enroll_owner.py).")
        return False, None

    record_to_file(verify_wav, VERIFY_SECONDS)
    energy = compute_energy(verify_wav)
    print(f"Detected energy = {energy:.6f}")
    if energy < ENERGY_THRESHOLD:
        print("No speech detected (too quiet). Try again closer to mic or increase volume.")
        return False, None

    owner_emb = np.load(EMBED_PATH)
    test_emb = get_embedding_from_file(verify_wav)
    # flatten if needed
    sim = 1.0 - cosine(owner_emb.flatten(), test_emb.flatten())
    return (sim >= SIMILARITY_THRESHOLD), sim

def transcribe_whisper(wav_path):
    """Run Whisper offline on a WAV and return the transcribed lowercase string."""
    model = get_whisper()
    result = model.transcribe(wav_path, language="en")
    return result.get("text","").strip().lower()

def execute_command_from_text(text):
    """Find an action in APP_MAP matching the text and execute it."""
    for key, action in APP_MAP.items():
        if key in text:
            kind = action.get("type")
            cmd = action.get("cmd")
            try:
                if kind == "system":
                    print(f"Executing system command: {cmd}")
                    os.system(cmd)
                elif kind == "path":
                    print(f"Launching binary: {cmd}")
                    subprocess.Popen([cmd], shell=False)
                elif kind == "url":
                    print(f"Opening URL: {cmd}")
                    webbrowser.open(cmd)
                else:
                    print("Unknown action type.")
                return True
            except Exception as e:
                print("Failed to execute action:", e)
                return False
    print("No matching app command found in your spoken text.")
    return False

# ---- Main flow ----
def main():
    print("EchoEdge — Owner-only voice command (offline)")
    print("Step 1: Verify owner voice")
    owner_ok, sim = verify_owner()
    if not owner_ok:
        print(f"Access Denied (similarity={sim}).")
        return

    print(f"Owner verified (similarity={sim:.3f}). Now speak your command.")
    time.sleep(0.5)
    cmd_wav = "command.wav"
    record_to_file(cmd_wav, COMMAND_SECONDS)
    energy = compute_energy(cmd_wav)
    print(f"Command energy = {energy:.6f}")
    if energy < ENERGY_THRESHOLD:
        print("No command detected (too quiet). Aborting.")
        return

    print("Transcribing command (Whisper)...")
    text = transcribe_whisper(cmd_wav)
    print("You said:", repr(text))
    if not text:
        print("Transcription empty.")
        return

    # Execute mapped app
    ok = execute_command_from_text(text)
    if ok:
        print("Action executed.")
    else:
        print("No action executed.")

if __name__ == "__main__":
    main()
