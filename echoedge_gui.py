# echoedge_gui.py
import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np
import sounddevice as sd
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
import whisper

FS = 16000
RECORD_TIME = 3
OWNER_FILE = "owner_embedding.npy"

# ---------- Core Functions ----------
def record_audio(filename, duration=RECORD_TIME):
    data = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, data, FS)
    return filename

def verify_owner(wav_path):
    if not os.path.exists(OWNER_FILE):
        messagebox.showwarning("Warning", "Please enroll owner first!")
        return False, None
    enc = VoiceEncoder()
    owner_emb = np.load(OWNER_FILE)
    wav = preprocess_wav(wav_path)
    emb = enc.embed_utterance(wav)
    sim = 1 - cosine(emb, owner_emb)
    return sim >= 0.65, sim

def transcribe_audio(wav_path):
    model = whisper.load_model("small")
    result = model.transcribe(wav_path)
    return result["text"]

# ---------- GUI ----------
class EchoEdgeApp:
    def __init__(self, root):
        self.root = root
        root.title("EchoEdge")
        root.geometry("420x560")
        root.configure(bg="#B32791")

        title = tk.Label(root, text="EchoEdge", font=("Helvetica", 28, "bold"), bg="#B32791", fg="white")
        title.pack(pady=(40,20))

        # Buttons
        btn_voice = tk.Button(root, text="🎙 Voice Control", font=("Arial", 14), bg="#C084FC", fg="white",
                              width=20, height=2, command=self.start_voice)
        btn_voice.pack(pady=15)

        btn_chat = tk.Button(root, text="💬 Chat", font=("Arial", 14), bg="#F87171", fg="white",
                             width=20, height=2, command=self.show_message)
        btn_chat.pack(pady=15)

        self.log = scrolledtext.ScrolledText(root, width=48, height=15, wrap=tk.WORD, font=("Arial", 11))
        self.log.pack(padx=10, pady=20)
        self.log.configure(state='disabled')

    def append(self, txt):
        self.log.configure(state='normal')
        self.log.insert(tk.END, txt + "\n")
        self.log.configure(state='disabled')
        self.log.see(tk.END)

    def show_message(self):
        messagebox.showinfo("Chat", "Chat function placeholder (future expansion).")

    def start_voice(self):
        threading.Thread(target=self.process_voice, daemon=True).start()

    def process_voice(self):
        try:
            self.append("Recording 3 s — speak now...")
            temp_wav = record_audio("temp.wav")
            owner_ok, sim = verify_owner(temp_wav)
            self.append(f"Similarity = {sim:.3f}")

            if not owner_ok:
                self.append("❌ Access Denied — Voice not recognized.")
                return

            self.append("✅ Voice recognized as Owner — transcribing...")
            text = transcribe_audio(temp_wav)
            self.append("🗣 You said: " + text)

        except Exception as e:
            messagebox.showerror("Error", str(e))

# ---------- Run ----------
if __name__ == "__main__":
    root = tk.Tk()
    EchoEdgeApp(root)
    root.mainloop()
