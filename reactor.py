"""
reactor.py — AI Song Reactor Bot (TikTok Live Edition)
Uses Whisper (local) + Librosa + Groq to react to a song for TikTok Live.

Run locally (mic):  python reactor.py
Run locally (file): python reactor.py song.mp3
Hosted (gunicorn / Render): gunicorn reactor:app
  POST an MP3 via multipart form to /react  (field name: "audio")
  curl -X POST https://your-app.onrender.com/react -F "audio=@song.mp3"
"""

import sys
import os
import json
import tempfile
import urllib.parse
import urllib.request
import whisper
import librosa
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from flask import Flask, request, jsonify
from groq import Groq

# ---------------------------------------------------------------------------
# CONFIG — edit these before running locally
# ---------------------------------------------------------------------------
RECORD_SECONDS = 45              # How many seconds to record from the microphone
SAMPLE_RATE = 44100              # Standard audio sample rate
GROQ_API_KEY = "YOUR_GROQ_API_KEY"  # Set via GROQ_API_KEY env var instead
WHISPER_MODEL = "tiny"           # tiny = fits Render free 512 MB RAM; use base locally
GROQ_MODEL = "llama3-8b-8192"    # Free Groq model (very fast)
# ---------------------------------------------------------------------------

TIKTOK_SYSTEM_PROMPT = """
You are a hyper-energetic TikTok Live host reacting to songs in real time.
Your audience is watching RIGHT NOW — so be loud, be fun, use emojis, and keep it SHORT.
When given a song's BPM and lyrics:
- Shout out the vibe and energy in one punchy sentence.
- Pick 1–2 standout lyric lines and dramatically react to them.
- Rate the song with a hype score like "9/10 CERTIFIED BANGER 🔥" or "5/10 mid but we vibe 😭".
- End with a call-to-action like "Drop a 🔥 if you felt this!" or "Smash that like if this SLAPS!".
Keep the whole reaction under 120 words. No long paragraphs. Maximum hype.
"""

# ---------------------------------------------------------------------------
# Flask app — created at module level so gunicorn can import it
# ---------------------------------------------------------------------------
app = Flask(__name__)


def _get_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", GROQ_API_KEY)


def search_wikimedia_images(query: str, limit: int = 8) -> list[dict]:
    """Find free-to-use themed images from Wikimedia Commons."""
    if not query.strip():
        return []

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",  # File namespace
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": "640",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    pages = payload.get("query", {}).get("pages", {})
    images = []
    for page in pages.values():
        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            continue
        info = imageinfo[0]
        images.append(
            {
                "title": page.get("title", "Untitled"),
                "url": info.get("thumburl") or info.get("url", ""),
            }
        )
    return images


def transcribe(audio_path: str) -> str:
    print(f"[1/3] Loading Whisper model '{WHISPER_MODEL}'...")
    model = whisper.load_model(WHISPER_MODEL)
    print(f"[1/3] Transcribing '{audio_path}'...")
    result = model.transcribe(audio_path)
    lyrics = result["text"].strip()
    print(f"      Done ({len(lyrics)} characters).\n")
    return lyrics


def detect_bpm(audio_path: str) -> float:
    print("[2/3] Analysing tempo with Librosa...")
    y, sr = librosa.load(audio_path, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(tempo), 1)
    print(f"      Detected BPM: {bpm}\n")
    return bpm


def get_groq_reaction(
    lyrics: str,
    bpm: float,
    api_key: str,
    image_title: str = "",
    image_url: str = "",
    preset_style: str = "",
) -> str:
    print("[3/3] Sending to Groq for a TikTok Live reaction...")
    client = Groq(api_key=api_key)

    visual_context = ""
    if preset_style:
        visual_context += (
            "Scene preset selected by host:\n"
            f"- Preset style: {preset_style}\n\n"
            "Match your delivery, vocabulary, and emotion to this preset style.\n\n"
        )

    if image_title or image_url:
        visual_context += (
            "Visual theme selected by host:\n"
            f"- Image title: {image_title or 'N/A'}\n"
            f"- Image URL: {image_url or 'N/A'}\n\n"
            "Align your reaction style and metaphors to this visual theme.\n\n"
        )

    prompt = (
        visual_context +
        f"BPM: {bpm}\n\n"
        f"Lyrics:\n{lyrics}\n\n"
        "React to this song for my TikTok Live RIGHT NOW!"
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": TIKTOK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
        return jsonify(
                {
                        "status": "Song Reactor Bot is live",
                        "usage": "Open /control for website UI, or POST /react with field 'audio'",
                }
        )


@app.route("/control", methods=["GET"])
def control_page():
        html = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Song Reactor Control</title>
    <style>
        :root {
            --bg1: #0f172a;
            --bg2: #1e293b;
            --card: #ffffff;
            --ink: #0f172a;
            --accent: #f97316;
            --muted: #64748b;
        }
        body {
            margin: 0;
            font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
            background: radial-gradient(circle at 20% 20%, #334155, var(--bg1));
            color: white;
            min-height: 100vh;
            padding: 24px;
        }
        .wrap {
            max-width: 980px;
            margin: 0 auto;
        }
        .card {
            background: var(--card);
            color: var(--ink);
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.25);
        }
        h1 { margin: 0 0 6px 0; }
        .sub { color: #e2e8f0; margin-bottom: 16px; }
        .row { display: flex; gap: 10px; flex-wrap: wrap; }
        .presetRow {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 10px 0 4px 0;
        }
        input[type=text], input[type=file] {
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 10px;
            font-size: 14px;
            flex: 1;
            min-width: 220px;
        }
        button {
            border: 0;
            border-radius: 10px;
            padding: 10px 14px;
            background: var(--accent);
            color: white;
            font-weight: 700;
            cursor: pointer;
        }
        .presetBtn {
            background: #e2e8f0;
            color: #0f172a;
        }
        .presetBtn.active {
            background: #0f172a;
            color: white;
        }
        #images {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 10px;
            margin-top: 12px;
        }
        .imgCard {
            border: 2px solid transparent;
            border-radius: 12px;
            overflow: hidden;
            background: #f8fafc;
            cursor: pointer;
        }
        .imgCard.selected { border-color: var(--accent); }
        .imgCard img { width: 100%; height: 120px; object-fit: cover; display: block; }
        .imgCard p {
            margin: 0;
            padding: 8px;
            font-size: 12px;
            color: var(--muted);
            min-height: 40px;
        }
        pre {
            white-space: pre-wrap;
            background: #f8fafc;
            border-radius: 12px;
            padding: 12px;
            border: 1px solid #e2e8f0;
        }
    </style>
</head>
<body>
    <div class="wrap">
        <h1>Song Reactor Control</h1>
        <div class="sub">Pick a theme image from online results, upload audio, then generate a reaction tied to that image.</div>

        <div class="card">
            <strong>Scene Presets</strong>
            <div id="presets" class="presetRow"></div>
            <p id="presetInfo">Preset: none</p>
            <div class="row">
                <input id="query" type="text" placeholder="Image theme, e.g. cyberpunk neon, rainy city, sunset beach" />
                <button onclick="searchImages()">Find Theme Images</button>
            </div>
            <div id="images"></div>
        </div>

        <div class="card">
            <div class="row">
                <input id="audio" type="file" accept="audio/*" />
                <button onclick="toggleMic()" id="micBtn">Record From Mic</button>
                <button onclick="reactNow()">Generate TikTok Reaction</button>
            </div>
            <p id="micStatus">Mic: idle</p>
            <p id="selectedInfo">Selected image: none</p>
            <pre id="output">Reaction output will appear here.</pre>
        </div>
    </div>

    <script>
        let selected = null;
        let mediaRecorder = null;
        let micChunks = [];
        let micBlob = null;
        let micStream = null;
        let selectedPreset = null;

        const PRESETS = [
            {
                name: 'Cyberpunk Night',
                query: 'cyberpunk city neon rain',
                style: 'futuristic neon chaos, fast and electric punchlines'
            },
            {
                name: 'Beach Sunset',
                query: 'sunset beach waves golden hour',
                style: 'warm summer chill, smooth flow, vibey and romantic'
            },
            {
                name: 'Anime Opening',
                query: 'anime sky action dramatic lighting',
                style: 'dramatic hero energy, cinematic intensity, motivational hype'
            },
            {
                name: 'Luxury Night',
                query: 'luxury penthouse skyline gold',
                style: 'high-status flex tone, confident and premium language'
            },
            {
                name: 'Haunted Mood',
                query: 'fog forest moon gothic',
                style: 'dark mysterious atmosphere, eerie but entertaining reactions'
            },
        ];

        function renderPresets() {
            const wrap = document.getElementById('presets');
            wrap.innerHTML = '';
            PRESETS.forEach((preset) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'presetBtn';
                btn.textContent = preset.name;
                btn.onclick = () => applyPreset(preset, btn);
                wrap.appendChild(btn);
            });
        }

        function applyPreset(preset, btnEl) {
            selectedPreset = preset;
            document.getElementById('query').value = preset.query;
            document.getElementById('presetInfo').textContent = 'Preset: ' + preset.name;
            document.querySelectorAll('.presetBtn').forEach((b) => b.classList.remove('active'));
            btnEl.classList.add('active');
            searchImages();
        }

        async function searchImages() {
            const q = document.getElementById('query').value.trim();
            if (!q) return;
            const res = await fetch('/images?query=' + encodeURIComponent(q));
            const data = await res.json();
            const container = document.getElementById('images');
            container.innerHTML = '';

            (data.images || []).forEach((img) => {
                const card = document.createElement('div');
                card.className = 'imgCard';
                card.innerHTML = `<img src="${img.url}" alt="${img.title}" /><p>${img.title}</p>`;
                card.onclick = () => {
                    selected = img;
                    document.querySelectorAll('.imgCard').forEach((x) => x.classList.remove('selected'));
                    card.classList.add('selected');
                    document.getElementById('selectedInfo').textContent = 'Selected image: ' + img.title;
                };
                container.appendChild(card);
            });
        }

        async function reactNow() {
            const audioInput = document.getElementById('audio');
            const pickedFile = audioInput.files.length ? audioInput.files[0] : null;
            const audioPayload = pickedFile || micBlob;

            if (!audioPayload) {
                alert('Please choose an audio file or record from mic first.');
                return;
            }

            const form = new FormData();
            form.append('audio', audioPayload, pickedFile ? pickedFile.name : 'mic_recording.webm');
            if (selected) {
                form.append('image_title', selected.title);
                form.append('image_url', selected.url);
            }
            if (selectedPreset) {
                form.append('preset_style', selectedPreset.style);
            }

            document.getElementById('output').textContent = 'Processing...';
            const res = await fetch('/react', { method: 'POST', body: form });
            const data = await res.json();

            if (!res.ok) {
                document.getElementById('output').textContent = 'Error: ' + (data.error || 'unknown error');
                return;
            }

            document.getElementById('output').textContent =
                `BPM: ${data.bpm}\n\nReaction:\n${data.reaction}\n\nLyrics:\n${data.lyrics}`;
        }

        async function toggleMic() {
            const btn = document.getElementById('micBtn');
            const status = document.getElementById('micStatus');

            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                try {
                    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(micStream);
                    micChunks = [];
                    micBlob = null;

                    mediaRecorder.ondataavailable = (evt) => {
                        if (evt.data && evt.data.size > 0) {
                            micChunks.push(evt.data);
                        }
                    };


                renderPresets();
                    mediaRecorder.onstop = () => {
                        micBlob = new Blob(micChunks, { type: 'audio/webm' });
                        status.textContent = 'Mic: recording saved in memory';
                        btn.textContent = 'Record From Mic';
                        if (micStream) {
                            micStream.getTracks().forEach((t) => t.stop());
                        }
                    };

                    mediaRecorder.start();
                    status.textContent = 'Mic: recording... click again to stop';
                    btn.textContent = 'Stop Recording';
                } catch (err) {
                    status.textContent = 'Mic error: ' + err;
                }
            } else if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }
    </script>
</body>
</html>
"""
        return html


@app.route("/images", methods=["GET"])
def image_search():
        query = request.args.get("query", "")
        images = search_wikimedia_images(query)
        return jsonify({"query": query, "images": images})


@app.route("/react", methods=["POST"])
def react():
    api_key = _get_api_key()
    if api_key == "YOUR_GROQ_API_KEY":
        return jsonify({"error": "GROQ_API_KEY not configured on server"}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file field found in request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to a temporary file so Whisper and Librosa can read it
    suffix = os.path.splitext(audio_file.filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    image_title = request.form.get("image_title", "")
    image_url = request.form.get("image_url", "")
    preset_style = request.form.get("preset_style", "")

    try:
        lyrics = transcribe(tmp_path)
        bpm = detect_bpm(tmp_path)
        reaction = get_groq_reaction(lyrics, bpm, api_key, image_title, image_url, preset_style)
        return jsonify({"bpm": bpm, "lyrics": lyrics, "reaction": reaction})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.unlink(tmp_path)  # Always clean up the temp file


# ---------------------------------------------------------------------------
# Microphone recording
# ---------------------------------------------------------------------------

def record_from_mic(duration: int = RECORD_SECONDS) -> str:
    print(f"[MIC] Recording for {duration} seconds... 🎤 Play your song now!")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
    )
    sd.wait()  # Block until recording is done
    print("[MIC] Recording complete.\n")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, audio)
    return tmp.name


# ---------------------------------------------------------------------------
# CLI entry point (local use only)
# ---------------------------------------------------------------------------

def main():
    api_key = _get_api_key()
    if api_key == "YOUR_GROQ_API_KEY":
        print("Error: Set GROQ_API_KEY env var or edit reactor.py.")
        print("Free key at https://console.groq.com")
        sys.exit(1)

    # If a file path is passed as an argument, use it; otherwise record from mic
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.isfile(audio_path):
            print(f"Error: File '{audio_path}' not found.")
            sys.exit(1)
        cleanup = False
    else:
        audio_path = record_from_mic()
        cleanup = True  # Temp file — delete after processing

    try:
        lyrics = transcribe(audio_path)
        bpm = detect_bpm(audio_path)
        reaction = get_groq_reaction(lyrics, bpm, api_key)
    finally:
        if cleanup:
            os.unlink(audio_path)

    print("=" * 60)
    print("TIKTOK LIVE REACTION 🎵")
    print("=" * 60)
    print(reaction)
    print("=" * 60)


if __name__ == "__main__":
    # Running directly → CLI mode (mic or file)
    # Running via gunicorn → gunicorn imports `app` directly (main() is never called)
    main()
