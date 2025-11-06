"""
gui_main.py
EchoEdge Desktop (dark fullscreen GUI)

Features:
- Home: Voice Control (owner-only) + Voice->Text (copyable)
- Settings: Enroll owner, Replace owner, Delete owner, Change threshold
- Uses WAV recording, Wav2Vec2 embeddings for speaker verification (torchaudio),
  Whisper for transcription (offline), FFmpeg must be installed (on PATH).
- Drop-in: edit APP_MAP to add/remove apps/URLs.
"""

import os
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import tkinter.scrolledtext as scrolledtext
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import whisper
import webbrowser
import subprocess
from scipy.spatial.distance import cosine
import pyperclip  # ensure installed: pip install pyperclip

# ---------------- CONFIG ----------------
SAMPLERATE = 16000
VERIFY_SECONDS = 3
COMMAND_SECONDS = 4
TRANSCRIBE_SECONDS = 6
EMBED_PATH = "owner_embedding.npy"
OWNER_WAV = "owner.wav"

# default threshold (you can change in settings)
SIMILARITY_THRESHOLD = 0.80
ENERGY_THRESHOLD = 0.0002

# Whisper model name (offline)
WHISPER_MODEL = "small"  # tiny/base/small (tradeoff speed vs accuracy)

# Map words to actions (edit these paths for your machine)
APP_MAP = {
    "calculator": {"type": "system", "cmd": "calc"},
    "notepad": {"type": "system", "cmd": "notepad"},
    "paint": {"type": "system", "cmd": "mspaint"},
    "explorer": {"type": "system", "cmd": "explorer"},
    "chrome": {"type": "path", "cmd": r"C:\Program Files\Google\Chrome\Application\chrome.exe"},
    "youtube": {"type": "url", "cmd": "https://www.youtube.com/"},
    "google": {"type": "url", "cmd": "https://www.google.com/"},
    # Add more mappings for your machine here
}

# ------------- Lazy models -------------
_wav2vec_model = None
_whisper_model = None

def get_wav2vec_model():
    global _wav2vec_model
    if _wav2vec_model is None:
        _wav2vec_model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
    return _wav2vec_model

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model

# ------------- audio helpers -------------
def record_to_file(path, duration_seconds):
    data = sd.rec(int(duration_seconds * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='float32')
    sd.wait()
    sf.write(path, data, SAMPLERATE)
    return path

def compute_energy(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:,0]
    return float(np.mean(data.astype(np.float64) ** 2))

def get_embedding_from_file(path):
    model = get_wav2vec_model()
    wav, sr = torchaudio.load(path)
    if sr != SAMPLERATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLERATE)
    with torch.no_grad():
        feats = model.extract_features(wav)
        # make robust across torchaudio versions
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if isinstance(feats, list):
            feats = feats[0]
        if not isinstance(feats, torch.Tensor):
            raise TypeError(f"Unexpected features type: {type(feats)}")
        emb = feats.mean(dim=1).cpu().numpy()
    return emb

def transcribe_with_whisper(path):
    model = get_whisper_model()
    result = model.transcribe(path, language="en")
    return result.get("text", "").strip()

# ------------- core actions -------------
def enroll_owner_record(duration=6.0):
    """Record owner voice and save embedding."""
    record_to_file(OWNER_WAV, duration)
    model = get_wav2vec_model()
    emb = get_embedding_from_file(OWNER_WAV)
    np.save(EMBED_PATH, emb)
    return OWNER_WAV, EMBED_PATH

def delete_owner_embedding():
    if os.path.exists(EMBED_PATH):
        os.remove(EMBED_PATH)
    if os.path.exists(OWNER_WAV):
        os.remove(OWNER_WAV)

def verify_owner_once(duration=VERIFY_SECONDS, threshold=None):
    """Record short clip and verify against EMBED_PATH. Returns (bool, similarity or None)."""
    if threshold is None:
        threshold = app_state["threshold"]
    if not os.path.exists(EMBED_PATH):
        return False, None
    tmp = "verify_tmp.wav"
    record_to_file(tmp, duration)
    energy = compute_energy(tmp)
    if energy < app_state["energy_threshold"]:
        return False, None
    owner_emb = np.load(EMBED_PATH)
    test_emb = get_embedding_from_file(tmp)
    sim = 1.0 - cosine(owner_emb.flatten(), test_emb.flatten())
    return (sim >= threshold), float(sim)

def execute_action_from_text(text):
    text = text.lower()
    for key, action in APP_MAP.items():
        if key in text:
            kind = action.get("type")
            cmd = action.get("cmd")
            try:
                if kind == "system":
                    os.system(cmd)
                elif kind == "path":
                    subprocess.Popen([cmd], shell=False)
                elif kind == "url":
                    webbrowser.open(cmd)
                return True, key
            except Exception as e:
                return False, str(e)
    return False, None

# ------------- UI and app state -------------
app_state = {
    "threshold": SIMILARITY_THRESHOLD,
    "energy_threshold": ENERGY_THRESHOLD,
    "status": "Idle"
}

# Utility: thread wrapper to avoid blocking UI
def run_in_thread(fn):
    def wrapper(*a, **k):
        threading.Thread(target=lambda: fn(*a, **k), daemon=True).start()
    return wrapper

# ------------- GUI -------------
class EchoEdgeGUI:
    def __init__(self, root):
        self.root = root
        root.title("EchoEdge")
        root.attributes("-fullscreen", True)  # fullscreen
        root.configure(bg="#0b0f14")  # dark background

        # top header
        header = tk.Frame(root, bg="#0b0f14")
        header.pack(fill="x", pady=12)
        title = tk.Label(header, text="EchoEdge", font=("Helvetica", 30, "bold"), fg="white", bg="#0b0f14")
        title.pack()

        # main frame for pages
        self.page_frame = tk.Frame(root, bg="#0b0f14")
        self.page_frame.pack(expand=True, fill="both")

        # bottom nav
        nav = tk.Frame(root, bg="#081018", height=60)
        nav.pack(fill="x")
        btn_home = tk.Button(nav, text="Home", command=self.show_home, bg="#1b2630", fg="white", width=12, height=2)
        btn_home.pack(side="left", padx=12, pady=8)
        btn_settings = tk.Button(nav, text="Settings", command=self.show_settings, bg="#1b2630", fg="white", width=12, height=2)
        btn_settings.pack(side="left", padx=12, pady=8)
        btn_quit = tk.Button(nav, text="Quit", command=self.on_quit, bg="#6b2222", fg="white", width=10, height=2)
        btn_quit.pack(side="right", padx=12, pady=8)

        # status bar
        self.status_var = tk.StringVar(value="Idle")
        status_bar = tk.Label(root, textvariable=self.status_var, anchor="w", bg="#061018", fg="#bfcbd6")
        status_bar.pack(fill="x")

        # pages
        self.home_page = None
        self.settings_page = None
        self.voice_text_page = None

        self.show_home()

    def set_status(self, txt):
        app_state["status"] = txt
        self.status_var.set(txt)
        self.root.update_idletasks()

    def clear_page(self):
        for w in self.page_frame.winfo_children():
            w.destroy()

    def show_home(self):
        self.clear_page()
        f = tk.Frame(self.page_frame, bg="#0b0f14")
        f.pack(expand=True, fill="both")

        # two big tiles
        tile_frame = tk.Frame(f, bg="#0b0f14")
        tile_frame.pack(expand=True)

        btn_voice = tk.Button(tile_frame, text="🎙 Voice Control Laptop", font=("Arial", 24),
                              bg="#242a33", fg="white", width=28, height=6, command=self.on_voice_control)
        btn_voice.grid(row=0, column=0, padx=40, pady=40)

        btn_v2t = tk.Button(tile_frame, text="🔤 Voice → Text", font=("Arial", 24),
                              bg="#242a33", fg="white", width=28, height=6, command=self.on_voice_to_text_page)
        btn_v2t.grid(row=0, column=1, padx=40, pady=40)

        self.home_page = f

    def show_settings(self):
        self.clear_page()
        f = tk.Frame(self.page_frame, bg="#0b0f14")
        f.pack(expand=True, fill="both", padx=40, pady=20)

        # Enrollment group
        g = tk.LabelFrame(f, text="Owner Voice (Enrollment)", fg="white", bg="#0b0f14", labelanchor="n")
        g.pack(fill="x", pady=8)
        g.configure(font=("Arial",12))
        enroll_btn = tk.Button(g, text="Enroll / Replace Owner", command=self.on_enroll_replace, bg="#2b6b33", fg="white")
        enroll_btn.pack(side="left", padx=8, pady=8)
        delete_btn = tk.Button(g, text="Delete Owner", command=self.on_delete_owner, bg="#6b2222", fg="white")
        delete_btn.pack(side="left", padx=8, pady=8)

        # Threshold group
        thrf = tk.LabelFrame(f, text="Verification Settings", fg="white", bg="#0b0f14", labelanchor="n")
        thrf.pack(fill="x", pady=8)
        tk.Label(thrf, text="Similarity threshold (0.5 - 0.99):", bg="#0b0f14", fg="white").pack(side="left", padx=6)
        self.thresh_var = tk.DoubleVar(value=app_state["threshold"])
        thr_entry = tk.Entry(thrf, textvariable=self.thresh_var, width=8)
        thr_entry.pack(side="left", padx=6)
        apply_thresh = tk.Button(thrf, text="Apply", command=self.on_apply_threshold, bg="#2b6b33", fg="white")
        apply_thresh.pack(side="left", padx=6)

        # Energy sensitivity
        tk.Label(thrf, text="  Energy sensitivity (lower = more sensitive):", bg="#0b0f14", fg="white").pack(side="left", padx=6)
        self.energy_var = tk.DoubleVar(value=app_state["energy_threshold"])
        energy_entry = tk.Entry(thrf, textvariable=self.energy_var, width=8)
        energy_entry.pack(side="left", padx=6)
        apply_energy = tk.Button(thrf, text="Apply", command=self.on_apply_energy, bg="#2b6b33", fg="white")
        apply_energy.pack(side="left", padx=6)

        # info
        info = tk.Label(f, text="Notes: Enroll voice in quiet place. If verification fails, increase recording length in enroll step.", bg="#0b0f14", fg="#bfcbd6")
        info.pack(pady=10)

        self.settings_page = f

    def on_quit(self):
        if messagebox.askyesno("Quit", "Exit EchoEdge?"):
            self.root.destroy()

    # ------------- Actions that run long should be threaded -------------
    @run_in_thread
    def on_enroll_replace(self):
        try:
            self.set_status("Recording owner voice (6s)...")
            enroll_owner_record(duration=6.0)
            self.set_status("Owner enrolled (saved owner_embedding.npy).")
            messagebox.showinfo("Enroll", "Owner enrolled and saved.")
        except Exception as e:
            messagebox.showerror("Enroll error", str(e))
            self.set_status("Idle")

    @run_in_thread
    def on_delete_owner(self):
        if not os.path.exists(EMBED_PATH):
            messagebox.showinfo("Delete", "No owner enrolled.")
            return
        if messagebox.askyesno("Delete", "Delete owner embedding and voice?"):
            delete_owner_embedding()
            self.set_status("Owner deleted.")
            messagebox.showinfo("Delete", "Owner data deleted.")

    def on_apply_threshold(self):
        try:
            v = float(self.thresh_var.get())
            if not (0.5 <= v <= 0.99):
                raise ValueError("threshold out of range")
            app_state["threshold"] = v
            messagebox.showinfo("Threshold", f"Threshold set to {v:.2f}")
        except Exception as e:
            messagebox.showerror("Threshold error", str(e))

    def on_apply_energy(self):
        try:
            v = float(self.energy_var.get())
            app_state["energy_threshold"] = v
            messagebox.showinfo("Energy", f"Energy sensitivity set to {v:.6f}")
        except Exception as e:
            messagebox.showerror("Energy error", str(e))

    @run_in_thread
    def on_voice_control(self):
        try:
            self.set_status("Verifying owner voice...")
            ok, sim = verify_owner_once(duration=VERIFY_SECONDS)
            if sim is None:
                messagebox.showwarning("Verify", "No owner enrolled or audio too quiet.")
                self.set_status("Idle")
                return
            self.set_status(f"Similarity = {sim:.3f}")
            if not ok:
                messagebox.showwarning("Access Denied", f"Not owner (similarity={sim:.3f})")
                self.set_status("Idle")
                return

            # owner verified -> record command
            self.set_status("Owner verified: recording command...")
            cmd_file = "gui_command.wav"
            record_to_file(cmd_file, COMMAND_SECONDS)
            e = compute_energy(cmd_file)
            if e < app_state["energy_threshold"]:
                messagebox.showwarning("Command", "Command too quiet. Try again.")
                self.set_status("Idle")
                return

            self.set_status("Transcribing command...")
            txt = transcribe_with_whisper(cmd_file)
            self.set_status("Recognized: " + txt)
            if not txt:
                messagebox.showinfo("Command", "No speech transcribed.")
                self.set_status("Idle")
                return

            ok2, info = execute_action_from_text(txt)
            if ok2:
                messagebox.showinfo("Action", f"Executed: {info}")
            else:
                messagebox.showinfo("Action", "No matching app/command found.")
            self.set_status("Idle")

        except Exception as e:
            messagebox.showerror("Voice control error", str(e))
            self.set_status("Idle")

    def on_voice_to_text_page(self):
        self.clear_page()
        f = tk.Frame(self.page_frame, bg="#0b0f14")
        f.pack(expand=True, fill="both", padx=20, pady=20)

        title = tk.Label(f, text="Voice → Text", font=("Helvetica", 24), fg="white", bg="#0b0f14")
        title.pack(pady=8)
        instr = tk.Label(f, text="Click Record, speak, then press Transcribe.", fg="#bfcbd6", bg="#0b0f14")
        instr.pack(pady=6)

        # controls
        btn_frame = tk.Frame(f, bg="#0b0f14")
        btn_frame.pack(pady=8)
        rec_btn = tk.Button(btn_frame, text="Record (4s)", command=lambda: self.v2t_record(TRANSCRIBE_SECONDS), bg="#2b6b33", fg="white")
        rec_btn.grid(row=0, column=0, padx=6)
        trans_btn = tk.Button(btn_frame, text="Transcribe", command=self.v2t_transcribe, bg="#1b65a5", fg="white")
        trans_btn.grid(row=0, column=1, padx=6)
        copy_btn = tk.Button(btn_frame, text="Copy Text", command=self.v2t_copy, bg="#6b6b6b", fg="white")
        copy_btn.grid(row=0, column=2, padx=6)

        # output text area
        self.v2t_text = scrolledtext.ScrolledText(f, width=120, height=20, wrap=tk.WORD, font=("Arial", 14))
        self.v2t_text.pack(pady=12)

        back = tk.Button(f, text="Back", command=self.show_home, bg="#333333", fg="white")
        back.pack(pady=8)

        self.voice_text_page = f

    @run_in_thread
    def v2t_record(self, seconds):
        try:
            self.set_status("Recording for transcription...")
            record_to_file("v2t.wav", seconds)
            self.set_status("Recording complete.")
        except Exception as e:
            messagebox.showerror("Record error", str(e))
            self.set_status("Idle")

    @run_in_thread
    def v2t_transcribe(self):
        try:
            if not os.path.exists("v2t.wav"):
                messagebox.showwarning("No audio", "Please record first.")
                return
            self.set_status("Transcribing with Whisper...")
            t = transcribe_with_whisper("v2t.wav")
            self.v2t_text.delete("1.0", tk.END)
            self.v2t_text.insert(tk.END, t)
            self.set_status("Transcription complete.")
        except Exception as e:
            messagebox.showerror("Transcribe error", str(e))
            self.set_status("Idle")

    def v2t_copy(self):
        txt = self.v2t_text.get("1.0", tk.END).strip()
        if not txt:
            messagebox.showinfo("Copy", "No text to copy.")
            return
        pyperclip.copy(txt)
        messagebox.showinfo("Copy", "Text copied to clipboard.")

# ---------------- run ----------------
if __name__ == "__main__":
    # quick safety checks
    try:
        root = tk.Tk()
        app = EchoEdgeGUI(root)
        root.mainloop()
    except Exception as e:
        print("Fatal error:", e)
