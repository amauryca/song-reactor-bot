"""
reactor.py - AI Song Reactor Bot (Simple Edition)

Single button: Record 45 seconds → Get live reaction

Local CLI:
  python reactor.py            # records from local microphone

Hosted:
  gunicorn reactor:app

Website:
  Open /
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback

import librosa
import numpy as np
import whisper
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RECORD_SECONDS = 45
SAMPLE_RATE = 16000
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
WHISPER_MODEL = "tiny"
GROQ_MODEL = "llama3-8b-8192"
MAX_UPLOAD_MB = 20
# ---------------------------------------------------------------------------

REACTION_PROMPT = """You are a hyper-energetic TikTok Live host reacting to songs in real time.
Your audience is watching RIGHT NOW. Be loud, fun, spontaneous, and keep it SHORT (2-3 sentences max).
React with pure energy and excitement. Use emojis sparingly but effectively."""

app = Flask(__name__)
_whisper_model_cache = None
_groq_client_cache = None


def _get_whisper_model():
    global _whisper_model_cache
    if _whisper_model_cache is None:
        print(f"[INIT] Loading Whisper {WHISPER_MODEL} model...")
        _whisper_model_cache = whisper.load_model(WHISPER_MODEL)
    return _whisper_model_cache


def _get_groq_client():
    global _groq_client_cache
    if _groq_client_cache is None:
        print(f"[INIT] Creating Groq client...")
        from groq import Groq
        _groq_client_cache = Groq(api_key=GROQ_API_KEY)
    return _groq_client_cache


def _convert_audio_to_wav(input_path: str) -> str:
    """Convert any audio format to WAV using ffmpeg."""
    output_path = input_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path, "-y"],
            capture_output=True,
            timeout=30,
            check=True,
        )
        return output_path
    except Exception as e:
        print(f"[ERROR] Audio conversion failed: {e}")
        raise


def transcribe(audio_path: str) -> str:
    """Transcribe audio using Whisper."""
    try:
        if not audio_path.endswith(".wav"):
            print(f"[AUDIO] Converting to WAV...")
            audio_path = _convert_audio_to_wav(audio_path)
        
        model = _get_whisper_model()
        print(f"[TRANSCRIBE] Processing...")
        
        try:
            result = model.transcribe(
                audio_path,
                fp16=False,
                condition_on_previous_text=False,
                temperature=0.0,
                without_timestamps=True,
            )
        except TypeError:
            result = model.transcribe(audio_path, fp16=False, condition_on_previous_text=False, temperature=0.0)
        
        text = result.get("text", "").strip()
        print(f"[TRANSCRIBE] Got {len(text)} characters: {text[:100]}...")
        return text
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        print(traceback.format_exc())
        raise


def detect_bpm(audio_path: str) -> float:
    """Detect BPM using Librosa."""
    try:
        if not audio_path.endswith(".wav"):
            print(f"[AUDIO] Converting to WAV for BPM...")
            audio_path = _convert_audio_to_wav(audio_path)
        
        print(f"[BPM] Analyzing...")
        y, sr = librosa.load(audio_path, mono=True, sr=22050)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(tempo), 1)
        print(f"[BPM] Detected: {bpm}")
        return bpm
    except Exception as e:
        print(f"[ERROR] BPM detection failed: {e}")
        print(traceback.format_exc())
        return 0.0


def get_reaction(lyrics: str, bpm: float) -> str:
    """Generate reaction using Groq."""
    try:
        client = _get_groq_client()
        prompt = f"""{REACTION_PROMPT}

Song lyrics/content: {lyrics[:500]}
BPM: {bpm}

Now give your LIVE reaction in 2-3 sentences MAX:"""
        
        print(f"[GROQ] Calling with {len(lyrics)} chars, {bpm} BPM...")
        message = client.messages.create(
            model=GROQ_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        reaction = message.content[0].text.strip()
        print(f"[GROQ] Got reaction: {reaction[:100]}...")
        return reaction
    except Exception as e:
        print(f"[ERROR] Groq call failed: {e}")
        print(traceback.format_exc())
        raise


def _save_upload_to_temp(audio_file) -> str:
    """Save uploaded audio file to temp."""
    suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = tmp.name
        audio_file.save(path)
    
    file_size = os.path.getsize(path)
    print(f"[AUDIO] Saved {audio_file.filename} to {path} ({file_size} bytes)")
    
    if file_size < 1000:
        print(f"[WARNING] Audio file suspiciously small ({file_size} bytes)")
    
    return path


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def index():
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Song Reactor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 500px;
            width: 100%;
            text-align: center;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .btn-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        button {
            flex: 1;
            padding: 15px 30px;
            font-size: 1.1em;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        #recordBtn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        #recordBtn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        #recordBtn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        #recordBtn.recording {
            background: #ff6b6b;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .status {
            color: #666;
            margin: 20px 0;
            font-size: 1em;
            min-height: 20px;
        }
        
        .status.recording {
            color: #ff6b6b;
            font-weight: bold;
        }
        
        .status.processing {
            color: #667eea;
            font-weight: bold;
        }
        
        #output {
            background: #f5f5f5;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            min-height: 100px;
            text-align: left;
            white-space: pre-wrap;
            word-break: break-word;
            color: #333;
            display: none;
            font-family: monospace;
            font-size: 0.95em;
            line-height: 1.6;
        }
        
        .error {
            color: #ff6b6b;
        }
        
        .success {
            color: #51cf66;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 AI Song Reactor</h1>
        <p class="subtitle">Record 45 seconds of audio and get a live reaction!</p>
        
        <div class="btn-group">
            <button id="recordBtn" onclick="toggleRecord()">🎙️ Start Recording</button>
        </div>
        
        <p class="status" id="status"></p>
        <pre id="output"></pre>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let audioBlob = null;
        let micStream = null;
        let recordingTimer = null;
        let recordingSeconds = 0;

        async function toggleRecord() {
            const btn = document.getElementById('recordBtn');
            const status = document.getElementById('status');
            const output = document.getElementById('output');

            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                // Start recording
                try {
                    console.log('Requesting microphone access...');
                    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    
                    mediaRecorder = new MediaRecorder(micStream);
                    audioChunks = [];
                    audioBlob = null;

                    mediaRecorder.ondataavailable = (evt) => {
                        if (evt.data && evt.data.size > 0) {
                            audioChunks.push(evt.data);
                        }
                    };

                    mediaRecorder.onstop = async () => {
                        audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        console.log('Recording stopped. Blob size:', audioBlob.size);
                        
                        if (micStream) {
                            micStream.getTracks().forEach((track) => track.stop());
                        }
                        
                        // Send to server
                        await sendAudioToServer();
                        
                        // Reset button
                        btn.textContent = '🎙️ Start Recording';
                        btn.classList.remove('recording');
                        btn.disabled = false;
                        clearInterval(recordingTimer);
                    };

                    mediaRecorder.start();
                    btn.textContent = '⏹️ Stop Recording';
                    btn.classList.add('recording');
                    status.className = 'status recording';
                    status.textContent = 'Recording... (will stop at 45 seconds)';
                    output.style.display = 'none';
                    recordingSeconds = 0;

                    // Auto-stop after 45 seconds
                    recordingTimer = setInterval(() => {
                        recordingSeconds++;
                        status.textContent = `Recording... ${recordingSeconds}s / 45s`;
                        
                        if (recordingSeconds >= 45) {
                            clearInterval(recordingTimer);
                            mediaRecorder.stop();
                        }
                    }, 1000);

                } catch (err) {
                    console.error('Microphone error:', err);
                    status.className = 'status error';
                    status.textContent = '❌ Microphone error: ' + err.message;
                    output.style.display = 'block';
                    output.textContent = err.toString();
                }
            } else if (mediaRecorder.state === 'recording') {
                // Stop recording
                mediaRecorder.stop();
            }
        }

        async function sendAudioToServer() {
            const btn = document.getElementById('recordBtn');
            const status = document.getElementById('status');
            const output = document.getElementById('output');

            if (!audioBlob || audioBlob.size === 0) {
                status.className = 'status error';
                status.textContent = '❌ No audio recorded';
                return;
            }

            btn.disabled = true;
            status.className = 'status processing';
            status.textContent = '🔄 Processing your audio...';
            output.style.display = 'none';

            try {
                const form = new FormData();
                form.append('audio', audioBlob, 'recording.webm');

                console.log('Sending audio to server...');
                const response = await fetch('/react', {
                    method: 'POST',
                    body: form
                });

                console.log('Response status:', response.status);
                const data = await response.json();
                console.log('Response data:', data);

                if (!response.ok) {
                    throw new Error(data.error || 'Server error: ' + response.status);
                }

                // Display results
                output.style.display = 'block';
                output.innerHTML = `<span class="success">✅ Reaction Generated!</span>

BPM: ${data.bpm}

<strong>Reaction:</strong>
${data.reaction}

<strong>Lyrics/Content (transcribed):</strong>
${data.lyrics}`;

                status.className = 'status success';
                status.textContent = '✅ Reaction ready!';

            } catch (err) {
                console.error('Error:', err);
                status.className = 'status error';
                status.textContent = '❌ Error: ' + err.message;
                output.style.display = 'block';
                output.innerHTML = `<span class="error">Error Details:\n${err.toString()}</span>`;
            } finally {
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""
    return html


@app.route("/react", methods=["POST"])
def react_once():
    print("[REACT] One-shot reaction started")
    
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        print("[ERROR] GROQ_API_KEY not configured")
        return jsonify({"error": "GROQ_API_KEY not configured on server"}), 500

    if "audio" not in request.files:
        print("[ERROR] No audio file in request")
        return jsonify({"error": "No 'audio' file found"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        print("[ERROR] Empty filename")
        return jsonify({"error": "Empty filename"}), 400

    tmp_path = None
    try:
        tmp_path = _save_upload_to_temp(audio_file)
        
        print("[REACT] Transcribing...")
        lyrics = transcribe(tmp_path)
        print(f"[REACT] Got lyrics: {lyrics[:100]}...")
        
        print("[REACT] Detecting BPM...")
        bpm = detect_bpm(tmp_path)
        print(f"[REACT] BPM: {bpm}")
        
        print("[REACT] Generating reaction...")
        reaction = get_reaction(lyrics, bpm)
        print("[REACT] ✅ Done!")
        
        return jsonify({
            "bpm": bpm,
            "lyrics": lyrics,
            "reaction": reaction
        })
    
    except Exception as exc:
        print(f"[ERROR] Reaction failed: {exc}")
        print(traceback.format_exc())
        return jsonify({"error": str(exc)}), 500
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print(f"[CLEANUP] Removed {tmp_path}")
            except Exception as e:
                print(f"[WARNING] Failed to clean up: {e}")
        
        # Clean up converted WAV
        if tmp_path and os.path.exists(tmp_path + ".wav"):
            try:
                os.unlink(tmp_path + ".wav")
                print(f"[CLEANUP] Removed {tmp_path}.wav")
            except Exception as e:
                print(f"[WARNING] Failed to clean up WAV: {e}")


# ---------------------------------------------------------------------------
# CLI MODE (local recording)
# ---------------------------------------------------------------------------

def record_from_mic():
    """Record audio from microphone (local CLI mode only)."""
    try:
        import sounddevice
        import scipy.io.wavfile
    except ImportError:
        print("[ERROR] sounddevice/scipy not installed. Install with: pip install -r requirements.txt")
        sys.exit(1)

    print(f"[CLI] Recording {RECORD_SECONDS} seconds from microphone...")
    audio_data = sounddevice.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sounddevice.wait()
    
    tmp_file = "/tmp/cli_recording.wav"
    scipy.io.wavfile.write(tmp_file, SAMPLE_RATE, audio_data)
    print(f"[CLI] Saved to {tmp_file}")
    return tmp_file


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # CLI mode
    if len(sys.argv) > 1:
        # File or record mode
        input_file = sys.argv[1]
        if input_file == "record":
            input_file = record_from_mic()
        
        print(f"\n[CLI] Processing: {input_file}")
        
        try:
            lyrics = transcribe(input_file)
            bpm = detect_bpm(input_file)
            reaction = get_reaction(lyrics, bpm)
            
            print(f"\n{'='*60}")
            print(f"BPM: {bpm}")
            print(f"{'='*60}")
            print(f"Lyrics:\n{lyrics}")
            print(f"{'='*60}")
            print(f"Reaction:\n{reaction}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
    else:
        # Web server mode
        print("[INIT] Starting Song Reactor Bot...")
        print("[INIT] Open http://localhost:5000/")
        app.run(debug=False, host="0.0.0.0", port=5000)
