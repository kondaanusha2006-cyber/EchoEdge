from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Home route (browser)
@app.route('/')
def home():
    return "<h2>EchoEdge Voice Assistant</h2><p>Owner Voice Verification System is Running!</p>"

# Example API route for verifying owner (for now just a placeholder)
@app.route('/verify', methods=['POST'])
def verify_owner():
    # In your real project, you’ll record or upload audio, verify with your existing logic.
    return jsonify({"message": "Owner verification simulated successfully!"})

# Example API route for voice command
@app.route('/command', methods=['POST'])
def command():
    data = request.get_json()
    user_command = data.get("command", "")
    # Here you can connect your code to execute that command (like opening apps)
    return jsonify({"response": f"Executing: {user_command}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
