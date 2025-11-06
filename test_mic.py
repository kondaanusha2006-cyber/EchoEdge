import sounddevice as sd
import soundfile as sf

FS = 16000
DURATION = 3
OUTFILE = "test.wav"

print("Recording for", DURATION, "seconds... Speak now.")
data = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()
sf.write(OUTFILE, data, FS)
print("Saved", OUTFILE)
