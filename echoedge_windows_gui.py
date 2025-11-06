import os
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
import numpy as np
import sounddevice as sd
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
import whisper
import webbrowser

# ---- Settings ----
FS = 16000
RECORD_DURATION = 3.0
EMBED_PATH = "owner_embedding.npy"
THRESHOLD = 0.65
WHISPER_MODEL = "tiny"

# ---- Lazy-load models ----
encoder = VoiceEncoder()
whisper_model = whisper.load_model(WHISPER_MODEL)
owner_embedding = None

def load_owner_embedding():
    global owner_embedding
    if owner_embedding is None and os.path.exists(EMBED_PATH):
        owner_embedding = np.load(EMBED_PATH)

def record_audio(filename="temp.wav", duration=RECORD_DURATION):
    data = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, data, FS)
    return filename

def verify_owner(wav_path):
    load_owner_embedding()
    if owner_embedding is None:
        return False, None
    wav = preprocess_wav(wav_path)
    emb = encoder.embed_utterance(wav)
    sim = 1.0 - cosine(emb, owner_embedding)
    return sim >= THRESHOLD, sim

def transcribe_audio(wav_path):
    result = whisper_model.transcribe(wav_path)
    return result["text"].strip()

def run_command(text):
    if "open youtube" in text:
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube..."
    elif "open google" in text:
        webbrowser.open("https://www.google.com")
        return "Opening Google..."
    elif "calculator" in text:
        os.system("calc.exe")
        return "Opening Calculator..."
    else:
        return "No command matched."

# ---- GUI Logic ----
class EchoEdgeApp:
    def __init__(self, root):
        self.root = root
        root.title("EchoEdge - Voice Assistant")
        root.geometry("420x550")

        tk.Label(root, text="EchoEdge", font=("Helvetica", 24, "bold")).pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="🎤 Enroll Voice", width=20, command=self.enroll_owner).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="💬 Chat", width=20, command=self.chat_with_assistant).grid(row=0, column=1, padx=10)

        self.output = scrolledtext.ScrolledText(root, width=50, height=20, font=("Arial", 12))
        self.output.pack(padx=10, pady=10)
        self.output.configure(state='disabled')

        self.status = tk.Label(root, text="Status: Idle", anchor="w")
        self.status.pack(fill="x")

    def log(self, message):
        self.output.configure(state='normal')
        self.output.insert(tk.END, message + "\n")
        self.output.configure(state='disabled')
        self.output.see(tk.END)

    def set_status(self, msg):
        self.status.config(text="Status: " + msg)
        self.root.update_idletasks()

    def enroll_owner(self):
        def _enroll():
            self.set_status("Recording 4s voice...")
            data = sd.rec(int(4 * FS), samplerate=FS, channels=1, dtype='float32')
            sd.wait()
            sf.write("owner.wav", data, FS)
            wav = preprocess_wav("owner.wav")
            emb = encoder.embed_utterance(wav)
            np.save(EMBED_PATH, emb)
            self.log("✅ Owner voice enrolled.")
            self.set_status("Idle")
        threading.Thread(target=_enroll, daemon=True).start()

    def chat_with_assistant(self):
        def _chat():
            self.set_status("Recording...")
            record_audio("temp_chat.wav")
            self.set_status("Verifying owner...")
            verified, sim = verify_owner("temp_chat.wav")
            self.log(f"[Similarity: {sim:.3f}]")
            if not verified:
                self.log("❌ Access Denied: Not owner.")
                self.set_status("Idle")
                return

            self.set_status("Transcribing...")
            text = transcribe_audio("temp_chat.wav")
            self.log("🗣️ You: " + text)

            self.set_status("Running command...")
            response = run_command(text.lower())
            self.log("🤖 EchoEdge: " + response)
            self.set_status("Idle")
        threading.Thread(target=_chat, daemon=True).start()

# ---- Run App ----
if __name__ == "__main__":
    root = tk.Tk()
    app = EchoEdgeApp(root)
    root.mainloop()
