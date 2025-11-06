# chatbox_gui.py
import os
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.spatial.distance import cosine

SAMPLE_RATE = 16000
RECORD_SECONDS = 3.0
OWNER_EMBED_PATH = "owner_embedding.npy"
SIM_THRESHOLD = 0.65
WHISPER_MODEL = "tiny"

def load_owner_embedding():
    if not os.path.exists(OWNER_EMBED_PATH):
        return None
    return np.load(OWNER_EMBED_PATH)

def record_clip(path="temp_chat.wav", duration=RECORD_SECONDS):
    data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    sf.write(path, data, SAMPLE_RATE)
    return path

def compute_similarity(path, owner_emb):
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder()
    wav = preprocess_wav(path)
    emb = encoder.embed_utterance(wav)
    sim = 1.0 - cosine(emb, owner_emb)
    return float(sim)

def transcribe(path):
    import whisper
    model = whisper.load_model(WHISPER_MODEL)
    r = model.transcribe(path)
    return r.get("text", "").strip()

class ChatGUI(tk.Tk):
    def _init_(self):
        super()._init_()
        self.title("EchoEdge Chatbox")
        self.geometry("520x420")
        self.owner_emb = load_owner_embedding()

        top = tk.Frame(self)
        top.pack(pady=12)
        tk.Button(top, text="Enroll Owner", command=self.enroll).pack(side="left", padx=8)
        self.chat_btn = tk.Button(top, text="Chat (Voice)", command=self.on_chat)
        self.chat_btn.pack(side="left", padx=8)

        self.status = tk.Label(self, text="Idle")
        self.status.pack()

        self.log = scrolledtext.ScrolledText(self, height=18, width=62, state='disabled', wrap=tk.WORD)
        self.log.pack(padx=12, pady=8)

    def append(self, text):
        self.log.configure(state='normal')
        self.log.insert(tk.END, text + "\n")
        self.log.configure(state='disabled')
        self.log.see(tk.END)

    def set_status(self, text):
        self.status.config(text=text)
        self.update_idletasks()

    def enroll(self):
        # simple enrollment: call external script if present
        if os.path.exists("enroll_owner.py"):
            # run it in background
            def run_enroll():
                self.set_status("Enrolling: recording...")
                try:
                    import subprocess
                    subprocess.run(["python", "enroll_owner.py"], check=True)
                    self.owner_emb = load_owner_embedding()
                    self.append("Owner enrolled.")
                except Exception as e:
                    messagebox.showerror("Enroll error", str(e))
                finally:
                    self.set_status("Idle")
            threading.Thread(target=run_enroll, daemon=True).start()
        else:
            messagebox.showinfo("Missing", "enroll_owner.py not found. Run enrollment script first.")

    def on_chat(self):
        def work():
            if self.owner_emb is None:
                messagebox.showwarning("Not enrolled", "Owner embedding missing. Enroll owner first.")
                return
            self.set_status("Recording...")
            clip = record_clip()
            self.set_status("Verifying...")
            try:
                sim = compute_similarity(clip, self.owner_emb)
            except Exception as e:
                messagebox.showerror("Verify error", str(e))
                self.set_status("Idle")
                return
            self.append(f"[debug] similarity={sim:.3f}")
            if sim < SIM_THRESHOLD:
                self.append("Access denied: speaker not recognized.")
                self.set_status("Idle")
                return
            self.set_status("Transcribing...")
            try:
                txt = transcribe(clip)
                self.append("Owner: " + (txt if txt else "[empty]"))
            except Exception as e:
                messagebox.showerror("Transcription error", str(e))
            self.set_status("Idle")

        threading.Thread(target=work, daemon=True).start()

if _name_ == "_main_":
    app = ChatGUI()
    app.mainloop()