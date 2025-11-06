# echoedge_interface.py
# EchoEdge interface (Tkinter, dark, fullscreen)
# - Home: Voice Control & Voice->Text
# - Grid of many apps (offline or online)
# - Settings for enroll / delete / threshold
# NOTE: Edit APP_MAP paths for apps installed on your PC.

import os
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
import tkinter.scrolledtext as scrolledtext
import webbrowser
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import whisper
from scipy.spatial.distance import cosine
import pyperclip

# ---------------- CONFIG ----------------
SAMPLERATE = 16000
VERIFY_SECONDS = 3
COMMAND_SECONDS = 4
TRANSCRIBE_SECONDS = 6
EMBED_PATH = "owner_embedding.npy"
OWNER_WAV = "owner.wav"

# default thresholds (can change in Settings)
SIMILARITY_THRESHOLD = 0.80
ENERGY_THRESHOLD = 0.0002

WHISPER_MODEL = "small"   # "tiny"/"base"/"small"

# ---------------- APP MAP ----------------
# For "type":"path" put the full exe path for your machine.
# For "type":"url" put the target web URL.
# For "type":"system" put an OS shortcut (like "notepad", "calc", "explorer")
APP_MAP = {
    # Browsers
    "google chrome": {"type": "path", "cmd": r"C:\Program Files\Google\Chrome\Application\chrome.exe"},
    "edge": {"type": "path", "cmd": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"},
    "firefox": {"type": "path", "cmd": r"C:\Program Files\Mozilla Firefox\firefox.exe"},
    "opera": {"type": "path", "cmd": r"C:\Program Files\Opera\launcher.exe"},
    # Office / editors
    "word": {"type": "path", "cmd": r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"},
    "excel": {"type": "path", "cmd": r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"},
    "powerpoint": {"type": "path", "cmd": r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"},
    "vs code": {"type": "path", "cmd": r"C:\Users\prava\AppData\Local\Programs\Microsoft VS Code\Code.exe"},
    "notepad": {"type": "system", "cmd": "notepad"},
    "notepad++": {"type": "path", "cmd": r"C:\Program Files\Notepad++\notepad++.exe"},
    # Media / utilities
    "calculator": {"type": "system", "cmd": "calc"},
    "vlc": {"type": "path", "cmd": r"C:\Program Files\VideoLAN\VLC\vlc.exe"},
    "spotify": {"type": "path", "cmd": r"C:\Users\prava\AppData\Roaming\Spotify\Spotify.exe"},
    "photos": {"type": "system", "cmd": "explorer"},  # open file explorer
    "file explorer": {"type": "system", "cmd": "explorer"},
    "task manager": {"type": "system", "cmd": "taskmgr"},
    # Development
    "pycharm": {"type": "path", "cmd": r"C:\Program Files\JetBrains\PyCharm Community Edition 2023.1\bin\pycharm64.exe"},
    "android studio": {"type": "path", "cmd": r"C:\Program Files\Android\Android Studio\bin\studio64.exe"},
    "xampp": {"type": "path", "cmd": r"C:\xampp\xampp-control.exe"},
    # Communication / conferencing (some are web)
    "zoom": {"type": "path", "cmd": r"C:\Users\prava\AppData\Roaming\Zoom\bin\Zoom.exe"},
    "teams": {"type": "path", "cmd": r"C:\Program Files\Microsoft\Teams\current\Teams.exe"},
    "whatsapp": {"type": "path", "cmd": r"C:\Users\prava\AppData\Local\Programs\WhatsApp\WhatsApp.exe"},
    "gmail": {"type": "url", "cmd": "https://mail.google.com/"},
    # Online services
    "youtube": {"type": "url", "cmd": "https://www.youtube.com/"},
    "google": {"type": "url", "cmd": "https://www.google.com/"},
    "drive": {"type": "url", "cmd": "https://drive.google.com/"},
    "one drive": {"type": "url", "cmd": "https://onedrive.live.com/"},
    "dropbox": {"type": "url", "cmd": "https://www.dropbox.com/"},
    "netflix": {"type": "url", "cmd": "https://www.netflix.com/"},
    "prime": {"type": "url", "cmd": "https://www.primevideo.com/"},
    # Graphics / design
    "photoshop": {"type": "path", "cmd": r"C:\Program Files\Adobe\Adobe Photoshop 2023\Photoshop.exe"},
    "illustrator": {"type": "path", "cmd": r"C:\Program Files\Adobe\Adobe Illustrator 2023\Illustrator.exe"},
    "canva": {"type": "url", "cmd": "https://www.canva.com/"},
    "figma": {"type": "url", "cmd": "https://www.figma.com/"},
    # Archives
    "winrar": {"type": "path", "cmd": r"C:\Program Files\WinRAR\WinRAR.exe"},
    "7-zip": {"type": "path", "cmd": r"C:\Program Files\7-Zip\7zFM.exe"},
    # Add other items as needed...
}

# ----------------- models lazy -----------------
_wav2vec = None
_whisper = None

def get_wav2vec():
    global _wav2vec
    if _wav2vec is None:
        _wav2vec = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
    return _wav2vec

def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = whisper.load_model(WHISPER_MODEL)
    return _whisper

# ---------------- audio helpers ----------------
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
    model = get_wav2vec()
    wav, sr = torchaudio.load(path)
    if sr != SAMPLERATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLERATE)
    with torch.no_grad():
        feats = model.extract_features(wav)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if isinstance(feats, list):
            feats = feats[0]
        if not isinstance(feats, torch.Tensor):
            raise TypeError("unexpected features type")
        emb = feats.mean(dim=1).cpu().numpy()
    return emb

def transcribe_whisper(path):
    model = get_whisper()
    result = model.transcribe(path, language="en")
    return result.get("text","").strip().lower()

# ---------------- speaker / command core ----------------
def enroll_owner(duration=6.0):
    record_to_file(OWNER_WAV, duration)
    emb = get_embedding_from_file(OWNER_WAV)
    np.save(EMBED_PATH, emb)
    return True

def delete_owner():
    if os.path.exists(EMBED_PATH):
        os.remove(EMBED_PATH)
    if os.path.exists(OWNER_WAV):
        os.remove(OWNER_WAV)

def verify_owner(duration=VERIFY_SECONDS, threshold=None, energy_threshold=None):
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    if energy_threshold is None:
        energy_threshold = ENERGY_THRESHOLD
    if not os.path.exists(EMBED_PATH):
        return False, None
    tmp = "verify_gui.wav"
    record_to_file(tmp, duration)
    energy = compute_energy(tmp)
    if energy < energy_threshold:
        return False, None
    owner_emb = np.load(EMBED_PATH)
    test_emb = get_embedding_from_file(tmp)
    sim = 1.0 - cosine(owner_emb.flatten(), test_emb.flatten())
    return (sim >= threshold), float(sim)

def execute_app_from_text(text):
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

# ---------------- GUI ----------------
class EchoEdgeApp:
    def __init__(self, root):
        self.root = root
        root.title("EchoEdge")
        root.attributes("-fullscreen", True)
        root.configure(bg="#0b0f14")
        self.state = {"threshold": SIMILARITY_THRESHOLD, "energy": ENERGY_THRESHOLD}
        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#0b0f14")
        header.pack(fill="x", pady=10)
        tk.Label(header, text="EchoEdge", font=("Segoe UI", 28, "bold"), fg="white", bg="#0b0f14").pack()

        self.page = tk.Frame(self.root, bg="#0b0f14")
        self.page.pack(expand=True, fill="both")

        nav = tk.Frame(self.root, bg="#081018", height=64)
        nav.pack(fill="x")
        tk.Button(nav, text="Home", command=self.show_home, bg="#1b2630", fg="white", width=10).pack(side="left", padx=8, pady=8)
        tk.Button(nav, text="Settings", command=self.show_settings, bg="#1b2630", fg="white", width=10).pack(side="left", padx=8)
        tk.Button(nav, text="Quit", command=self._quit, bg="#6b2222", fg="white", width=10).pack(side="right", padx=12)

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(self.root, textvariable=self.status_var, bg="#061018", fg="#bfcbd6", anchor="w").pack(fill="x")

        self.show_home()

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def clear(self):
        for w in self.page.winfo_children():
            w.destroy()

    # --- Home page ---
    def show_home(self):
        self.clear()
        f = tk.Frame(self.page, bg="#0b0f14")
        f.pack(expand=True, fill="both")

        # top tiles
        tiles = tk.Frame(f, bg="#0b0f14")
        tiles.pack(pady=24)
        t1 = self._create_tile(tiles, "🎙 Voice Control Laptop", self.start_voice_control)
        t2 = self._create_tile(tiles, "🔤 Voice → Text", self.show_v2t_page)
        t1.grid(row=0, column=0, padx=40, pady=10)
        t2.grid(row=0, column=1, padx=40, pady=10)

        # app grid (scrollable)
        grid_frame = tk.Frame(f, bg="#0b0f14")
        grid_frame.pack(expand=True, fill="both", padx=40, pady=10)
        self._build_app_grid(grid_frame)

    def _create_tile(self, parent, text, cmd):
        b = tk.Button(parent, text=text, width=28, height=6, bg="#242a33", fg="white", font=("Segoe UI", 16), command=lambda: threading.Thread(target=cmd, daemon=True).start())
        b.bind("<Enter>", lambda e: b.configure(bg="#2f3942"))
        b.bind("<Leave>", lambda e: b.configure(bg="#242a33"))
        return b

    def _build_app_grid(self, parent):
        # create a simple grid of buttons for keys in APP_MAP
        keys = list(APP_MAP.keys())
        rows = (len(keys) + 3) // 4
        r = 0; c = 0
        for i, key in enumerate(keys):
            btn = tk.Button(parent, text=key.title(), width=22, height=3, bg="#1c2328", fg="white", command=lambda k=key: threading.Thread(target=self._open_app_click, args=(k,), daemon=True).start())
            btn.grid(row=r, column=c, padx=8, pady=8)
            btn.bind("<Enter>", lambda e,b=btn: b.configure(bg="#2b3338"))
            btn.bind("<Leave>", lambda e,b=btn: b.configure(bg="#1c2328"))
            c += 1
            if c >= 4:
                c = 0; r += 1

    def _open_app_click(self, key):
        self.set_status(f"Opening {key} ...")
        action = APP_MAP.get(key)
        if not action:
            messagebox.showinfo("Open", f"No mapping for {key}")
            self.set_status("Idle")
            return
        kind = action["type"]; cmd = action["cmd"]
        try:
            if kind == "system":
                os.system(cmd)
            elif kind == "path":
                subprocess.Popen([cmd], shell=False)
            elif kind == "url":
                webbrowser.open(cmd)
            self.set_status("Idle")
        except Exception as e:
            messagebox.showerror("Open error", str(e))
            self.set_status("Idle")

    # --- Voice control flow ---
    def start_voice_control(self):
        # Verify owner
        self.set_status("Verifying owner voice...")
        if not os.path.exists(EMBED_PATH):
            messagebox.showwarning("Verify", "No owner enrolled. Go to Settings → Enroll Owner.")
            self.set_status("Idle")
            return
        ok, sim = verify_owner(duration=VERIFY_SECONDS, threshold=self.state["threshold"], energy_threshold=self.state["energy"])
        if sim is None:
            messagebox.showwarning("Verify", "No speech or too quiet.")
            self.set_status("Idle")
            return
        if not ok:
            messagebox.showwarning("Access Denied", f"Not owner (similarity={sim:.3f})")
            self.set_status("Idle")
            return
        self.set_status(f"Owner verified ({sim:.3f}). Recording command...")
        cmdfile = "gui_cmd.wav"
        record_to_file(cmdfile, COMMAND_SECONDS)
        e = compute_energy(cmdfile)
        if e < self.state["energy"]:
            messagebox.showwarning("Command", "Command too quiet.")
            self.set_status("Idle")
            return
        self.set_status("Transcribing...")
        txt = transcribe_whisper(cmdfile)
        if not txt:
            messagebox.showinfo("Transcribe", "No text recognized.")
            self.set_status("Idle")
            return
        self.set_status("Executing: " + txt)
        ok2, info = execute_app_from_text(txt)
        if ok2:
            messagebox.showinfo("Action", f"Executed: {info}")
        else:
            messagebox.showinfo("Action", "No matching app found in your spoken text.")
        self.set_status("Idle")

    # --- Voice → Text page ---
    def show_v2t_page(self):
        self.clear()
        f = tk.Frame(self.page, bg="#0b0f14")
        f.pack(expand=True, fill="both", padx=20, pady=20)
        tk.Label(f, text="Voice → Text", font=("Segoe UI", 22), fg="white", bg="#0b0f14").pack(pady=8)
        btn_frame = tk.Frame(f, bg="#0b0f14"); btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Record (6s)", bg="#2b6b33", fg="white", command=lambda: threading.Thread(target=self._v2t_record, daemon=True).start()).grid(row=0, column=0, padx=8)
        tk.Button(btn_frame, text="Transcribe", bg="#1b65a5", fg="white", command=lambda: threading.Thread(target=self._v2t_transcribe, daemon=True).start()).grid(row=0, column=1, padx=8)
        tk.Button(btn_frame, text="Copy", bg="#6b6b6b", fg="white", command=self._v2t_copy).grid(row=0, column=2, padx=8)
        self.v2t_area = scrolledtext.ScrolledText(f, width=120, height=20, font=("Segoe UI", 14))
        self.v2t_area.pack(pady=12)
        tk.Button(f, text="Back", command=self.show_home, bg="#333333", fg="white").pack(pady=8)

    def _v2t_record(self):
        self.set_status("Recording for v2t...")
        record_to_file("v2t.wav", TRANSCRIBE_SECONDS)
        self.set_status("Recorded v2t.wav")

    def _v2t_transcribe(self):
        if not os.path.exists("v2t.wav"):
            messagebox.showwarning("No audio", "Record first.")
            return
        self.set_status("Transcribing...")
        t = transcribe_whisper("v2t.wav")
        self.v2t_area.delete("1.0", tk.END)
        self.v2t_area.insert(tk.END, t)
        self.set_status("Transcription done.")

    def _v2t_copy(self):
        txt = self.v2t_area.get("1.0", tk.END).strip()
        if txt:
            pyperclip.copy(txt)
            messagebox.showinfo("Copy", "Copied to clipboard.")
        else:
            messagebox.showinfo("Copy", "No text to copy.")

    # --- Settings ---
    def show_settings(self):
        self.clear()
        f = tk.Frame(self.page, bg="#0b0f14"); f.pack(expand=True, fill="both", padx=20, pady=20)
        tk.Label(f, text="Settings", font=("Segoe UI", 20), fg="white", bg="#0b0f14").pack(pady=8)

        # enroll / replace
        b1 = tk.Button(f, text="Enroll / Replace Owner (6s)", bg="#2b6b33", fg="white", command=lambda: threading.Thread(target=self._enroll_owner, daemon=True).start())
        b1.pack(pady=6)
        b2 = tk.Button(f, text="Delete Owner", bg="#6b2222", fg="white", command=lambda: threading.Thread(target=self._delete_owner, daemon=True).start())
        b2.pack(pady=6)

        # threshold
        thrf = tk.Frame(f, bg="#0b0f14"); thrf.pack(pady=8)
        tk.Label(thrf, text="Similarity threshold (0.5-0.99):", fg="white", bg="#0b0f14").pack(side="left", padx=6)
        self.thr_var = tk.DoubleVar(value=self.state["threshold"])
        tk.Entry(thrf, textvariable=self.thr_var, width=8).pack(side="left", padx=6)
        tk.Button(thrf, text="Apply", bg="#2b6b33", fg="white", command=self._apply_threshold).pack(side="left", padx=6)

        # energy
        enf = tk.Frame(f, bg="#0b0f14"); enf.pack(pady=8)
        tk.Label(enf, text="Energy sensitivity (lower = more sensitive):", fg="white", bg="#0b0f14").pack(side="left", padx=6)
        self.energy_var = tk.DoubleVar(value=self.state["energy"])
        tk.Entry(enf, textvariable=self.energy_var, width=10).pack(side="left", padx=6)
        tk.Button(enf, text="Apply", bg="#2b6b33", fg="white", command=self._apply_energy).pack(side="left", padx=6)

        tk.Button(f, text="Back", bg="#333333", fg="white", command=self.show_home).pack(pady=12)

    def _apply_threshold(self):
        v = float(self.thr_var.get())
        if not (0.5 <= v <= 0.99):
            messagebox.showerror("Error", "Threshold out of range.")
            return
        self.state["threshold"] = v
        messagebox.showinfo("Threshold", f"Set to {v:.2f}")

    def _apply_energy(self):
        v = float(self.energy_var.get())
        self.state["energy"] = v
        messagebox.showinfo("Energy", f"Set to {v:.6f}")

    def _enroll_owner(self):
        try:
            self.set_status("Recording owner voice (6s)...")
            enroll_owner(duration=6.0)
            self.set_status("Owner enrolled.")
            messagebox.showinfo("Enroll", "Owner enrolled successfully.")
        except Exception as e:
            messagebox.showerror("Enroll error", str(e))
            self.set_status("Idle")

    def _delete_owner(self):
        if not os.path.exists(EMBED_PATH):
            messagebox.showinfo("Delete", "No owner enrolled.")
            return
        if messagebox.askyesno("Delete", "Delete owner data?"):
            delete_owner()
            messagebox.showinfo("Delete", "Owner removed.")
            self.set_status("Idle")

    def _quit(self):
        if messagebox.askyesno("Exit", "Do you want to exit?"):
            self.root.destroy()

# ---------------- run ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EchoEdgeApp(root)
    root.mainloop()
