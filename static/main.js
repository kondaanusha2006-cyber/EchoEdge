async function recordAndUpload(endpoint, resultElemId) {
  const resultElem = document.getElementById(resultElemId);
  resultElem.textContent = "Requesting microphone...";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const chunks = [];

    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/wav' });
      const form = new FormData();
      form.append('file', blob, 'speech.wav');

      resultElem.textContent = "Uploading...";
      const resp = await fetch(endpoint, { method: 'POST', body: form });
      const data = await resp.json();
      resultElem.textContent = JSON.stringify(data);
      stream.getTracks().forEach(t => t.stop());
    };

    // record for 3 seconds:
    mediaRecorder.start();
    setTimeout(() => mediaRecorder.stop(), 3000);
  } catch (err) {
    resultElem.textContent = "Microphone error: " + err;
  }
}

// wire buttons
document.getElementById('record-control').addEventListener('click', () => {
  recordAndUpload('/api/voice-control', 'control-result');
});
document.getElementById('record-transcribe').addEventListener('click', () => {
  recordAndUpload('/api/transcribe', 'transcription');
});
