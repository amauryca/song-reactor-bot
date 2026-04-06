"""
reactor.py - AI Song Reactor Bot (TikTok Live Edition)

Local CLI:
  python reactor.py            # records from local microphone
  python reactor.py song.mp3   # uses local file

Hosted:
  gunicorn reactor:app

Website control:
  Open /control
"""

import json
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from uuid import uuid4

import librosa
import numpy as np
import whisper
from flask import Flask, jsonify, request
from groq import Groq

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RECORD_SECONDS = 45
SAMPLE_RATE = 44100
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
WHISPER_MODEL = "tiny"  # tiny keeps memory low on free hosting tiers
GROQ_MODEL = "llama3-8b-8192"
MAX_LIVE_LYRICS_CHARS = 900
LIVE_CHUNK_MS = 2500
LIVE_BPM_EVERY_N_CHUNKS = 2
MAX_UPLOAD_MB = 20
LIVE_SESSION_TTL_SECONDS = 15 * 60
MAX_LIVE_SESSIONS = 12
# ---------------------------------------------------------------------------

TIKTOK_SYSTEM_PROMPT = """
You are a hyper-energetic TikTok Live host reacting to songs in real time.
Your audience is watching RIGHT NOW, so be loud, fun, and short.
When given BPM and lyrics:
- Call out the vibe and energy in one punchy sentence.
- Pick 1-2 standout lyric moments and react dramatically.
- Rate it with a hype score like "9/10 CERTIFIED BANGER".
- End with a call-to-action like "Drop a FIRE in chat".
Keep it under 120 words.
""".strip()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_WHISPER_MODEL_CACHE = None
LIVE_SESSIONS = {}


def _get_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", GROQ_API_KEY)


def _get_whisper_model():
    global _WHISPER_MODEL_CACHE
    if _WHISPER_MODEL_CACHE is None:
        print(f"[INIT] Loading Whisper model '{WHISPER_MODEL}'...")
        _WHISPER_MODEL_CACHE = whisper.load_model(WHISPER_MODEL)
    return _WHISPER_MODEL_CACHE


    def _cleanup_live_sessions() -> None:
      now = time.time()
      expired = [
        session_id
        for session_id, session in LIVE_SESSIONS.items()
        if now - session.get("updated_at", now) > LIVE_SESSION_TTL_SECONDS
      ]
      for session_id in expired:
        LIVE_SESSIONS.pop(session_id, None)

      if len(LIVE_SESSIONS) > MAX_LIVE_SESSIONS:
        oldest = sorted(
          LIVE_SESSIONS.items(),
          key=lambda item: item[1].get("updated_at", 0),
        )
        for session_id, _ in oldest[: len(LIVE_SESSIONS) - MAX_LIVE_SESSIONS]:
          LIVE_SESSIONS.pop(session_id, None)


def search_wikimedia_images(query: str, limit: int = 8) -> list[dict]:
    if not query.strip():
        return []

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
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
        info_list = page.get("imageinfo", [])
        if not info_list:
            continue
        info = info_list[0]
        images.append(
            {
                "title": page.get("title", "Untitled"),
                "url": info.get("thumburl") or info.get("url", ""),
            }
        )
    return images


def transcribe(audio_path: str) -> str:
    model = _get_whisper_model()
    # Prefer low-latency decoding options for short live chunks.
    try:
        result = model.transcribe(
            audio_path,
            fp16=False,
            condition_on_previous_text=False,
            temperature=0.0,
            without_timestamps=True,
        )
    except TypeError:
        # Fallback for older whisper builds without without_timestamps.
        result = model.transcribe(
            audio_path,
            fp16=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
    return result.get("text", "").strip()


def detect_bpm(audio_path: str) -> float:
    # Keep BPM analysis lighter by downsampling for live chunks.
    y, sr = librosa.load(audio_path, mono=True, sr=22050)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(float(tempo), 1)


def _merge_lyrics(existing: str, chunk_text: str) -> str:
    merged = (existing + " " + chunk_text).strip()
    if len(merged) > MAX_LIVE_LYRICS_CHARS:
        return merged[-MAX_LIVE_LYRICS_CHARS:]
    return merged


def get_groq_reaction(
    lyrics: str,
    bpm: float,
    api_key: str,
    image_title: str = "",
    image_url: str = "",
    preset_style: str = "",
    live_hint: str = "",
    live_mode: bool = False,
) -> str:
    client = Groq(api_key=api_key)

    context = ""
    if preset_style:
        context += (
            "Scene preset selected by host:\n"
            f"- Preset style: {preset_style}\n\n"
            "Match your delivery and vocabulary to this preset.\n\n"
        )

    if image_title or image_url:
        context += (
            "Visual theme selected by host:\n"
            f"- Image title: {image_title or 'N/A'}\n"
            f"- Image URL: {image_url or 'N/A'}\n\n"
            "Align your metaphors and mood to this visual.\n\n"
        )

    if live_hint:
        context += f"Live context: {live_hint}\n\n"

    prompt = (
        context
        + f"BPM: {bpm}\n\n"
        + f"Lyrics:\n{lyrics}\n\n"
        + "React to this song for TikTok Live right now."
    )

    max_tokens = 220
    if live_mode:
        prompt += "\n\nKeep this update ultra-short (2-3 lines)."
        max_tokens = 110

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": TIKTOK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()


def _save_upload_to_temp(audio_file) -> str:
    suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = tmp.name
        audio_file.save(path)
    return path


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "status": "Song Reactor Bot is live",
            "usage": "Open /control for website UI",
        }
    )


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"ok": True, "live_sessions": len(LIVE_SESSIONS)})


@app.route("/control", methods=["GET"])
def control_page():
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Song Reactor Control</title>
  <style>
    :root {{
      --bg1: #0f172a;
      --bg2: #1e293b;
      --card: #ffffff;
      --ink: #0f172a;
      --accent: #f97316;
      --muted: #64748b;
      --ok: #16a34a;
      --warn: #dc2626;
    }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: radial-gradient(circle at 20% 20%, #334155, var(--bg1));
      color: white;
      min-height: 100vh;
      padding: 24px;
    }}
    .wrap {{ max-width: 1040px; margin: 0 auto; }}
    .card {{
      background: var(--card);
      color: var(--ink);
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.25);
    }}
    h1 {{ margin: 0 0 6px 0; }}
    .sub {{ color: #e2e8f0; margin-bottom: 16px; }}
    .row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .presetRow {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 4px 0; }}
    input[type=text], input[type=file] {{
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 10px;
      font-size: 14px;
      flex: 1;
      min-width: 220px;
    }}
    button {{
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }}
    .presetBtn {{ background: #e2e8f0; color: #0f172a; }}
    .presetBtn.active {{ background: #0f172a; color: white; }}
    .liveBtn {{ background: #0f172a; }}
    .stopBtn {{ background: #dc2626; }}
    .statusOk {{ color: var(--ok); font-weight: 700; }}
    .statusWarn {{ color: var(--warn); font-weight: 700; }}
    #images {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .imgCard {{
      border: 2px solid transparent;
      border-radius: 12px;
      overflow: hidden;
      background: #f8fafc;
      cursor: pointer;
    }}
    .imgCard.selected {{ border-color: var(--accent); }}
    .imgCard img {{ width: 100%; height: 120px; object-fit: cover; display: block; }}
    .imgCard p {{ margin: 0; padding: 8px; font-size: 12px; color: var(--muted); min-height: 40px; }}
    pre {{
      white-space: pre-wrap;
      background: #f8fafc;
      border-radius: 12px;
      padding: 12px;
      border: 1px solid #e2e8f0;
      min-height: 130px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Song Reactor Control</h1>
    <div class="sub">Use one-shot mode or start LIVE mode for rolling real-time reactions.</div>

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
      <h3>One-Shot Reaction</h3>
      <div class="row">
        <input id="audio" type="file" accept="audio/*" />
        <button onclick="toggleMic()" id="micBtn">Record Short Clip</button>
        <button onclick="reactNow()">Generate One-Shot Reaction</button>
      </div>
      <p id="micStatus">Mic: idle</p>
    </div>

    <div class="card">
      <h3>LIVE Reaction (continuous)</h3>
      <div class="row">
        <button class="liveBtn" onclick="startLive()">Start Live Mode</button>
        <button class="stopBtn" onclick="stopLive()">Stop Live Mode</button>
      </div>
      <p id="liveStatus" class="statusWarn">Live: stopped</p>
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

    let liveSessionId = null;
    let liveUploading = false;

    const PRESETS = [
      {{ name: 'Cyberpunk Night', query: 'cyberpunk city neon rain', style: 'futuristic neon chaos, fast and electric punchlines' }},
      {{ name: 'Beach Sunset', query: 'sunset beach waves golden hour', style: 'warm summer chill, smooth flow, vibey and romantic' }},
      {{ name: 'Anime Opening', query: 'anime sky action dramatic lighting', style: 'dramatic hero energy, cinematic intensity, motivational hype' }},
      {{ name: 'Luxury Night', query: 'luxury penthouse skyline gold', style: 'high-status flex tone, confident and premium language' }},
      {{ name: 'Haunted Mood', query: 'fog forest moon gothic', style: 'dark mysterious atmosphere, eerie but entertaining reactions' }},
    ];

    function renderPresets() {{
      const wrap = document.getElementById('presets');
      wrap.innerHTML = '';
      PRESETS.forEach((preset) => {{
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'presetBtn';
        btn.textContent = preset.name;
        btn.onclick = () => applyPreset(preset, btn);
        wrap.appendChild(btn);
      }});
    }}

    function applyPreset(preset, btnEl) {{
      selectedPreset = preset;
      document.getElementById('query').value = preset.query;
      document.getElementById('presetInfo').textContent = 'Preset: ' + preset.name;
      document.querySelectorAll('.presetBtn').forEach((b) => b.classList.remove('active'));
      btnEl.classList.add('active');
      searchImages();
    }}

    async function searchImages() {{
      const q = document.getElementById('query').value.trim();
      if (!q) return;
      const res = await fetch('/images?query=' + encodeURIComponent(q));
      const data = await res.json();
      const container = document.getElementById('images');
      container.innerHTML = '';

      (data.images || []).forEach((img) => {{
        const card = document.createElement('div');
        card.className = 'imgCard';
        card.innerHTML = `<img src="${{img.url}}" alt="${{img.title}}" /><p>${{img.title}}</p>`;
        card.onclick = () => {{
          selected = img;
          document.querySelectorAll('.imgCard').forEach((x) => x.classList.remove('selected'));
          card.classList.add('selected');
          document.getElementById('selectedInfo').textContent = 'Selected image: ' + img.title;
        }};
        container.appendChild(card);
      }});
    }}

    function appendStyleContext(form) {{
      if (selected) {{
        form.append('image_title', selected.title);
        form.append('image_url', selected.url);
      }}
      if (selectedPreset) {{
        form.append('preset_style', selectedPreset.style);
      }}
    }}

    async function reactNow() {{
      const audioInput = document.getElementById('audio');
      const pickedFile = audioInput.files.length ? audioInput.files[0] : null;
      const audioPayload = pickedFile || micBlob;
      if (!audioPayload) {{
        alert('Please choose an audio file or record from mic first.');
        return;
      }}

      const form = new FormData();
      form.append('audio', audioPayload, pickedFile ? pickedFile.name : 'mic_recording.webm');
      appendStyleContext(form);

      document.getElementById('output').textContent = 'Processing one-shot...';
      const res = await fetch('/react', {{ method: 'POST', body: form }});
      const data = await res.json();
      if (!res.ok) {{
        document.getElementById('output').textContent = 'Error: ' + (data.error || 'unknown error');
        return;
      }}

      document.getElementById('output').textContent =
        `BPM: ${{data.bpm}}\n\nReaction:\n${{data.reaction}}\n\nLyrics:\n${{data.lyrics}}`;
    }}

    async function toggleMic() {{
      const btn = document.getElementById('micBtn');
      const status = document.getElementById('micStatus');

      if (!mediaRecorder || mediaRecorder.state === 'inactive') {{
        try {{
          micStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
          mediaRecorder = new MediaRecorder(micStream);
          micChunks = [];
          micBlob = null;

          mediaRecorder.ondataavailable = (evt) => {{
            if (evt.data && evt.data.size > 0) micChunks.push(evt.data);
          }};

          mediaRecorder.onstop = () => {{
            micBlob = new Blob(micChunks, {{ type: 'audio/webm' }});
            status.textContent = 'Mic: short clip ready';
            btn.textContent = 'Record Short Clip';
            if (micStream) micStream.getTracks().forEach((t) => t.stop());
          }};

          mediaRecorder.start();
          status.textContent = 'Mic: recording short clip... click again to stop';
          btn.textContent = 'Stop Recording';
        }} catch (err) {{
          status.textContent = 'Mic error: ' + err;
        }}
      }} else if (mediaRecorder.state === 'recording') {{
        mediaRecorder.stop();
      }}
    }}

    async function startLive() {{
      if (liveSessionId) return;
      const liveStatus = document.getElementById('liveStatus');

      const startRes = await fetch('/live/start', {{ method: 'POST' }});
      const startData = await startRes.json();
      if (!startRes.ok) {{
        liveStatus.className = 'statusWarn';
        liveStatus.textContent = 'Live start failed: ' + (startData.error || 'unknown error');
        return;
      }}

      liveSessionId = startData.session_id;

      try {{
        micStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
        mediaRecorder = new MediaRecorder(micStream, {{ mimeType: 'audio/webm' }});
        mediaRecorder.ondataavailable = async (evt) => {{
          if (!evt.data || evt.data.size === 0 || !liveSessionId || liveUploading) return;
          liveUploading = true;
          try {{
            const form = new FormData();
            form.append('session_id', liveSessionId);
            form.append('audio', evt.data, 'live_chunk.webm');
            appendStyleContext(form);

            const res = await fetch('/live/chunk', {{ method: 'POST', body: form }});
            const data = await res.json();
            if (!res.ok) {{
              document.getElementById('output').textContent = 'Live error: ' + (data.error || 'unknown error');
              return;
            }}

            document.getElementById('output').textContent =
              `LIVE chunk #${{data.chunk_index}}\n` +
              `Current BPM: ${{data.bpm_current}} | Avg BPM: ${{data.bpm_avg}}\n\n` +
              `Reaction:\n${{data.reaction}}\n\n` +
              `Lyrics so far:\n${{data.lyrics_preview}}`;
          }} finally {{
            liveUploading = false;
          }}
        }};

        mediaRecorder.start({LIVE_CHUNK_MS});
        liveStatus.className = 'statusOk';
        liveStatus.textContent = 'Live: running (sending chunks every {LIVE_CHUNK_MS/1000:.0f}s)';
      }} catch (err) {{
        await stopLive();
        liveStatus.className = 'statusWarn';
        liveStatus.textContent = 'Live mic error: ' + err;
      }}
    }}

    async function stopLive() {{
      const liveStatus = document.getElementById('liveStatus');
      if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
      if (micStream) micStream.getTracks().forEach((t) => t.stop());
      micStream = null;

      if (liveSessionId) {{
        const res = await fetch('/live/stop', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ session_id: liveSessionId }})
        }});
        const data = await res.json();
        if (res.ok && data.final_reaction) {{
          document.getElementById('output').textContent =
            `LIVE FINAL\nAvg BPM: ${{data.bpm_avg}}\n\nReaction:\n${{data.final_reaction}}`;
        }}
      }}

      liveSessionId = null;
      liveStatus.className = 'statusWarn';
      liveStatus.textContent = 'Live: stopped';
    }}

    renderPresets();
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
def react_once():
    api_key = _get_api_key()
    if api_key == "YOUR_GROQ_API_KEY":
        return jsonify({"error": "GROQ_API_KEY not configured on server"}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file field found in request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_title = request.form.get("image_title", "")
    image_url = request.form.get("image_url", "")
    preset_style = request.form.get("preset_style", "")

    tmp_path = _save_upload_to_temp(audio_file)
    try:
        lyrics = transcribe(tmp_path)
        bpm = detect_bpm(tmp_path)
        reaction = get_groq_reaction(lyrics, bpm, api_key, image_title, image_url, preset_style)
        return jsonify({"bpm": bpm, "lyrics": lyrics, "reaction": reaction})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/live/start", methods=["POST"])
def live_start():
    api_key = _get_api_key()
    if api_key == "YOUR_GROQ_API_KEY":
        return jsonify({"error": "GROQ_API_KEY not configured on server"}), 500

    _cleanup_live_sessions()
    session_id = str(uuid4())
    LIVE_SESSIONS[session_id] = {
        "lyrics": "",
        "bpm_values": [],
        "chunk_index": 0,
        "last_bpm_current": 0.0,
        "image_title": "",
        "image_url": "",
        "preset_style": "",
        "last_reaction": "",
        "updated_at": time.time(),
    }
    return jsonify({"session_id": session_id})


@app.route("/live/chunk", methods=["POST"])
def live_chunk():
    api_key = _get_api_key()
    _cleanup_live_sessions()
    session_id = request.form.get("session_id", "")
    session = LIVE_SESSIONS.get(session_id)
    if not session:
        return jsonify({"error": "Invalid or expired live session"}), 400

    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' in live chunk"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty live audio chunk"}), 400

    # Keep latest style controls from the web panel.
    session["image_title"] = request.form.get("image_title", session["image_title"])
    session["image_url"] = request.form.get("image_url", session["image_url"])
    session["preset_style"] = request.form.get("preset_style", session["preset_style"])
    session["updated_at"] = time.time()

    tmp_path = _save_upload_to_temp(audio_file)
    try:
        chunk_lyrics = transcribe(tmp_path)
        session["chunk_index"] += 1

        # Compute BPM every N chunks to reduce CPU load in live mode.
        should_refresh_bpm = (
            session["chunk_index"] == 1
            or session["chunk_index"] % LIVE_BPM_EVERY_N_CHUNKS == 0
        )
        if should_refresh_bpm:
            bpm_current = detect_bpm(tmp_path)
            session["last_bpm_current"] = bpm_current
        else:
            bpm_current = session.get("last_bpm_current", 0.0)

        if bpm_current > 0:
            session["bpm_values"].append(bpm_current)

        session["lyrics"] = _merge_lyrics(session["lyrics"], chunk_lyrics)

        bpm_avg = round(sum(session["bpm_values"]) / len(session["bpm_values"]), 1)
        live_hint = f"Chunk #{session['chunk_index']} in an ongoing live stream. React as if this is happening now."
        reaction = get_groq_reaction(
            session["lyrics"],
            bpm_avg,
            api_key,
            session["image_title"],
            session["image_url"],
            session["preset_style"],
            live_hint=live_hint,
            live_mode=True,
        )
        session["last_reaction"] = reaction

        return jsonify(
            {
                "session_id": session_id,
                "chunk_index": session["chunk_index"],
                "bpm_current": bpm_current,
                "bpm_avg": bpm_avg,
                "lyrics_preview": session["lyrics"],
                "reaction": reaction,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/live/stop", methods=["POST"])
def live_stop():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id", "")
    session = LIVE_SESSIONS.pop(session_id, None)
    if not session:
        return jsonify({"error": "Invalid or expired live session"}), 400

    bpm_values = session.get("bpm_values", [])
    bpm_avg = round(sum(bpm_values) / len(bpm_values), 1) if bpm_values else 0.0
    return jsonify(
        {
            "session_id": session_id,
            "chunks": session.get("chunk_index", 0),
            "bpm_avg": bpm_avg,
            "final_reaction": session.get("last_reaction", ""),
        }
    )


# ---------------------------------------------------------------------------
# Local microphone helper for CLI mode
# ---------------------------------------------------------------------------

def record_from_mic(duration: int = RECORD_SECONDS) -> str:
  try:
    import scipy.io.wavfile as wav
    import sounddevice as sd
  except Exception as exc:
    print("Local microphone mode requires sounddevice and PortAudio.")
    print("Install local deps with: pip install sounddevice scipy")
    print(f"Underlying error: {exc}")
    sys.exit(1)

    print(f"[MIC] Recording for {duration} seconds. Play your song now...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
    )
    sd.wait()
    print("[MIC] Recording complete.")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, audio)
    return tmp.name


# ---------------------------------------------------------------------------
# CLI entry point (local use only)
# ---------------------------------------------------------------------------

def main():
    api_key = _get_api_key()
    if api_key == "YOUR_GROQ_API_KEY":
        print("Error: set GROQ_API_KEY env var or edit reactor.py")
        sys.exit(1)

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.isfile(audio_path):
            print(f"Error: file '{audio_path}' not found")
            sys.exit(1)
        cleanup = False
    else:
        audio_path = record_from_mic()
        cleanup = True

    try:
        lyrics = transcribe(audio_path)
        bpm = detect_bpm(audio_path)
        reaction = get_groq_reaction(lyrics, bpm, api_key)
    finally:
        if cleanup:
            os.unlink(audio_path)

    print("=" * 60)
    print("TIKTOK LIVE REACTION")
    print("=" * 60)
    print(reaction)
    print("=" * 60)


if __name__ == "__main__":
    main()
