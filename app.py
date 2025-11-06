"""
EchoEdge Web App (Flask version)

Features:
- Home: Voice Control (owner-only) + Voice->Text (copyable)
- Settings: Enroll owner, Replace owner, Delete owner, Change threshold
- Uses WAV recording, Wav2Vec2 embeddings for speaker verification (torchaudio),
  Whisper for transcription (offline), FFmpeg must be installed (on PATH).
- Drop-in: edit APP_MAP to add/remove apps/URLs.
"""

import os
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import whisper
import webbrowser
import subprocess
from scipy.spatial.distance import cosine
import json
import base64
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_file

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

# ------------- App state -------------
app_state = {
    "threshold": SIMILARITY_THRESHOLD,
    "energy_threshold": ENERGY_THRESHOLD,
    "status": "Idle"
}

# ------------- Flask App -------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', 
                         app_state=app_state,
                         owner_enrolled=os.path.exists(EMBED_PATH))

@app.route('/api/enroll_owner', methods=['POST'])
def api_enroll_owner():
    try:
        enroll_owner_record(duration=6.0)
        return jsonify({"status": "success", "message": "Owner enrolled successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/delete_owner', methods=['POST'])
def api_delete_owner():
    try:
        if not os.path.exists(EMBED_PATH):
            return jsonify({"status": "error", "message": "No owner enrolled"}), 400
        delete_owner_embedding()
        return jsonify({"status": "success", "message": "Owner data deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/update_threshold', methods=['POST'])
def api_update_threshold():
    try:
        threshold = float(request.json.get('threshold'))
        if not (0.5 <= threshold <= 0.99):
            raise ValueError("Threshold must be between 0.5 and 0.99")
        app_state["threshold"] = threshold
        return jsonify({"status": "success", "message": f"Threshold updated to {threshold:.2f}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/update_energy_threshold', methods=['POST'])
def api_update_energy_threshold():
    try:
        energy_threshold = float(request.json.get('energy_threshold'))
        app_state["energy_threshold"] = energy_threshold
        return jsonify({"status": "success", "message": f"Energy threshold updated to {energy_threshold:.6f}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/voice_control', methods=['POST'])
def api_voice_control():
    try:
        # Verify owner
        ok, sim = verify_owner_once(duration=VERIFY_SECONDS)
        if sim is None:
            return jsonify({"status": "error", "message": "No owner enrolled or audio too quiet"}), 400
        
        if not ok:
            return jsonify({"status": "denied", "message": f"Access denied (similarity={sim:.3f})", "similarity": sim})
        
        # Record command
        cmd_file = "web_command.wav"
        record_to_file(cmd_file, COMMAND_SECONDS)
        e = compute_energy(cmd_file)
        if e < app_state["energy_threshold"]:
            return jsonify({"status": "error", "message": "Command too quiet. Try again."}), 400
        
        # Transcribe command
        txt = transcribe_with_whisper(cmd_file)
        if not txt:
            return jsonify({"status": "error", "message": "No speech transcribed"}), 400
        
        # Execute action
        ok2, info = execute_action_from_text(txt)
        if ok2:
            return jsonify({
                "status": "success", 
                "message": f"Executed: {info}",
                "transcription": txt,
                "similarity": sim
            })
        else:
            return jsonify({
                "status": "no_match", 
                "message": "No matching app/command found",
                "transcription": txt,
                "similarity": sim
            })
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/voice_to_text', methods=['POST'])
def api_voice_to_text():
    try:
        # Record audio
        v2t_file = "web_v2t.wav"
        record_to_file(v2t_file, TRANSCRIBE_SECONDS)
        
        # Transcribe
        txt = transcribe_with_whisper(v2t_file)
        
        return jsonify({
            "status": "success", 
            "transcription": txt
        })
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/status')
def api_status():
    return jsonify({
        "app_state": app_state,
        "owner_enrolled": os.path.exists(EMBED_PATH)
    })

def create_html_template():
    """Create the HTML template with proper encoding"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EchoEdge Web</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0b0f14;
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid #1b2630;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            color: white;
            margin-bottom: 10px;
        }
        
        .nav {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
            padding: 15px;
            background: #081018;
            border-radius: 10px;
        }
        
        .nav button {
            padding: 12px 24px;
            background: #1b2630;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        
        .nav button:hover {
            background: #2b3640;
        }
        
        .nav button.quit {
            background: #6b2222;
        }
        
        .nav button.quit:hover {
            background: #8b3232;
        }
        
        .page {
            display: none;
            padding: 20px;
        }
        
        .page.active {
            display: block;
        }
        
        .tile-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }
        
        .tile {
            background: #242a33;
            padding: 40px 20px;
            border-radius: 15px;
            text-align: center;
            cursor: pointer;
            transition: transform 0.3s, background 0.3s;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .tile:hover {
            transform: translateY(-5px);
            background: #2b3640;
        }
        
        .tile h2 {
            font-size: 1.8rem;
            margin-bottom: 15px;
        }
        
        .tile p {
            color: #bfcbd6;
            font-size: 1.1rem;
        }
        
        .settings-section {
            background: #1b2630;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            border: 1px solid #2b3640;
        }
        
        .settings-section h3 {
            color: white;
            margin-bottom: 20px;
            font-size: 1.4rem;
            border-bottom: 1px solid #2b3640;
            padding-bottom: 10px;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        
        button.primary {
            background: #2b6b33;
            color: white;
        }
        
        button.primary:hover {
            background: #3b8b43;
        }
        
        button.danger {
            background: #6b2222;
            color: white;
        }
        
        button.danger:hover {
            background: #8b3232;
        }
        
        button.secondary {
            background: #1b65a5;
            color: white;
        }
        
        button.secondary:hover {
            background: #2b85c5;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #bfcbd6;
        }
        
        .form-group input {
            width: 200px;
            padding: 10px;
            border: 1px solid #2b3640;
            border-radius: 5px;
            background: #0b0f14;
            color: white;
            margin-right: 10px;
        }
        
        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #061018;
            padding: 15px;
            border-top: 1px solid #1b2630;
        }
        
        .status-text {
            color: #bfcbd6;
        }
        
        .transcription-box {
            background: #0b0f14;
            border: 1px solid #2b3640;
            border-radius: 5px;
            padding: 20px;
            min-height: 300px;
            color: white;
            font-size: 16px;
            line-height: 1.5;
            margin: 20px 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .alert {
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        
        .alert.success {
            background: #2b6b33;
            color: white;
        }
        
        .alert.error {
            background: #6b2222;
            color: white;
        }
        
        .alert.warning {
            background: #8b6b22;
            color: white;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #bfcbd6;
        }
        
        .spinner {
            border: 4px solid #2b3640;
            border-top: 4px solid #2b6b33;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>EchoEdge</h1>
            <p>Voice Control System</p>
        </div>
        
        <div class="nav">
            <button onclick="showPage('home')">Home</button>
            <button onclick="showPage('settings')">Settings</button>
            <button class="quit" onclick="window.close()">Quit</button>
        </div>
        
        <!-- Home Page -->
        <div id="home" class="page active">
            <div class="tile-grid">
                <div class="tile" onclick="voiceControl()">
                    <h2>Voice Control Laptop</h2>
                    <p>Owner-only voice commands to control your applications</p>
                </div>
                <div class="tile" onclick="showPage('voiceToText')">
                    <h2>Voice to Text</h2>
                    <p>Convert your speech to text and copy to clipboard</p>
                </div>
            </div>
        </div>
        
        <!-- Voice to Text Page -->
        <div id="voiceToText" class="page">
            <h2 style="text-align: center; margin-bottom: 20px;">Voice to Text</h2>
            <p style="text-align: center; color: #bfcbd6; margin-bottom: 30px;">
                Click Record, speak, then press Transcribe.
            </p>
            
            <div style="text-align: center; margin-bottom: 30px;">
                <button class="primary" onclick="recordForTranscription()">Record (6s)</button>
                <button class="secondary" onclick="transcribeAudio()">Transcribe</button>
                <button onclick="copyTranscription()">Copy Text</button>
                <button onclick="showPage('home')" style="background: #333; color: white;">Back</button>
            </div>
            
            <div class="transcription-box" id="transcriptionOutput">
                Transcription will appear here...
            </div>
            
            <div id="v2tLoading" class="loading">
                <div class="spinner"></div>
                <p>Processing...</p>
            </div>
        </div>
        
        <!-- Settings Page -->
        <div id="settings" class="page">
            <div class="settings-section">
                <h3>Owner Voice (Enrollment)</h3>
                <div class="button-group">
                    <button class="primary" onclick="enrollOwner()">Enroll / Replace Owner</button>
                    <button class="danger" onclick="deleteOwner()">Delete Owner</button>
                </div>
                <p style="color: #bfcbd6; margin-top: 15px; font-size: 14px;">
                    Owner enrolled: <span id="ownerStatus">{{ "Yes" if owner_enrolled else "No" }}</span>
                </p>
            </div>
            
            <div class="settings-section">
                <h3>Verification Settings</h3>
                <div class="form-group">
                    <label for="thresholdInput">Similarity threshold (0.5 - 0.99):</label>
                    <input type="number" id="thresholdInput" step="0.01" min="0.5" max="0.99" value="{{ app_state.threshold }}">
                    <button class="primary" onclick="updateThreshold()">Apply</button>
                </div>
                
                <div class="form-group">
                    <label for="energyInput">Energy sensitivity (lower = more sensitive):</label>
                    <input type="number" id="energyInput" step="0.000001" min="0.000001" value="{{ app_state.energy_threshold }}">
                    <button class="primary" onclick="updateEnergyThreshold()">Apply</button>
                </div>
            </div>
            
            <div style="color: #bfcbd6; padding: 20px; background: #0b0f14; border-radius: 5px;">
                <p><strong>Notes:</strong> Enroll voice in quiet place. If verification fails, increase recording length in enroll step.</p>
            </div>
        </div>
        
        <!-- Alert Container -->
        <div id="alertContainer" style="position: fixed; top: 20px; right: 20px; z-index: 1000;"></div>
    </div>
    
    <div class="status-bar">
        <div class="status-text" id="statusText">Idle</div>
    </div>

    <script>
        let currentTranscription = '';
        
        function showPage(pageId) {
            document.querySelectorAll('.page').forEach(page => {
                page.classList.remove('active');
            });
            document.getElementById(pageId).classList.add('active');
        }
        
        function setStatus(message) {
            document.getElementById('statusText').textContent = message;
        }
        
        function showAlert(message, type = 'info') {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert ${type}`;
            alertDiv.textContent = message;
            
            const container = document.getElementById('alertContainer');
            container.appendChild(alertDiv);
            
            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        }
        
        function showLoading(show) {
            document.getElementById('v2tLoading').style.display = show ? 'block' : 'none';
        }
        
        async function enrollOwner() {
            setStatus('Recording owner voice (6s)...');
            try {
                const response = await fetch('/api/enroll_owner', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    showAlert('Owner enrolled successfully', 'success');
                    document.getElementById('ownerStatus').textContent = 'Yes';
                } else {
                    showAlert('Error: ' + data.message, 'error');
                }
            } catch (error) {
                showAlert('Error: ' + error.message, 'error');
            }
            setStatus('Idle');
        }
        
        async function deleteOwner() {
            if (!confirm('Delete owner embedding and voice?')) return;
            
            try {
                const response = await fetch('/api/delete_owner', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    showAlert('Owner data deleted', 'success');
                    document.getElementById('ownerStatus').textContent = 'No';
                } else {
                    showAlert('Error: ' + data.message, 'error');
                }
            } catch (error) {
                showAlert('Error: ' + error.message, 'error');
            }
        }
        
        async function updateThreshold() {
            const threshold = parseFloat(document.getElementById('thresholdInput').value);
            
            try {
                const response = await fetch('/api/update_threshold', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ threshold })
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    showAlert(data.message, 'success');
                } else {
                    showAlert('Error: ' + data.message, 'error');
                }
            } catch (error) {
                showAlert('Error: ' + error.message, 'error');
            }
        }
        
        async function updateEnergyThreshold() {
            const energy_threshold = parseFloat(document.getElementById('energyInput').value);
            
            try {
                const response = await fetch('/api/update_energy_threshold', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ energy_threshold })
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    showAlert(data.message, 'success');
                } else {
                    showAlert('Error: ' + data.message, 'error');
                }
            } catch (error) {
                showAlert('Error: ' + error.message, 'error');
            }
        }
        
        async function voiceControl() {
            setStatus('Verifying owner voice...');
            
            try {
                const response = await fetch('/api/voice_control', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    showAlert(`Action executed: ${data.message}`, 'success');
                    setStatus(`Recognized: ${data.transcription}`);
                } else if (data.status === 'denied') {
                    showAlert(data.message, 'warning');
                    setStatus(`Access denied (similarity: ${data.similarity.toFixed(3)})`);
                } else if (data.status === 'no_match') {
                    showAlert(data.message, 'warning');
                    setStatus(`No match for: ${data.transcription}`);
                } else {
                    showAlert('Error: ' + data.message, 'error');
                    setStatus('Idle');
                }
            } catch (error) {
                showAlert('Error: ' + error.message, 'error');
                setStatus('Idle');
            }
        }
        
        async function recordForTranscription() {
            setStatus('Recording for transcription...');
            showAlert('Recording... Speak now.', 'info');
            
            // In a real implementation, you'd use the Web Audio API
            // For this demo, we'll simulate the recording
            setTimeout(() => {
                setStatus('Recording complete');
                showAlert('Recording complete', 'success');
            }, 6000);
        }
        
        async function transcribeAudio() {
            setStatus('Transcribing with Whisper...');
            showLoading(true);
            
            try {
                const response = await fetch('/api/voice_to_text', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    currentTranscription = data.transcription;
                    document.getElementById('transcriptionOutput').textContent = data.transcription || 'No speech detected';
                    showAlert('Transcription complete', 'success');
                } else {
                    showAlert('Error: ' + data.message, 'error');
                }
            } catch (error) {
                showAlert('Error: ' + error.message, 'error');
            }
            
            showLoading(false);
            setStatus('Idle');
        }
        
        function copyTranscription() {
            if (!currentTranscription) {
                showAlert('No text to copy', 'warning');
                return;
            }
            
            navigator.clipboard.writeText(currentTranscription).then(() => {
                showAlert('Text copied to clipboard', 'success');
            }).catch(() => {
                showAlert('Failed to copy text', 'error');
            });
        }
        
        // Initialize
        showPage('home');
        setStatus('Ready');
    </script>
</body>
</html>'''
    return html_content

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template with proper encoding
    html_content = create_html_template()
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Starting EchoEdge Web Server...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
    
