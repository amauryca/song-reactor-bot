"""
reactor.py — AI Song Reactor Bot (TikTok Live Edition)
Uses Whisper (local) + Librosa + Groq to react to a song for TikTok Live.

Run locally:  python reactor.py song.mp3
Hosted (gunicorn / Render): gunicorn reactor:app
  POST an MP3 via multipart form to /react  (field name: "audio")
  curl -X POST https://your-app.onrender.com/react -F "audio=@song.mp3"
"""

import sys
import os
import tempfile
import whisper
import librosa
from flask import Flask, request, jsonify
from groq import Groq

# ---------------------------------------------------------------------------
# CONFIG — edit these before running locally
# ---------------------------------------------------------------------------
AUDIO_FILE = "song.mp3"          # Default local MP3 (CLI mode only)
GROQ_API_KEY = "YOUR_GROQ_API_KEY"  # Free key at console.groq.com
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


def get_groq_reaction(lyrics: str, bpm: float, api_key: str) -> str:
    print("[3/3] Sending to Groq for a TikTok Live reaction...")
    client = Groq(api_key=api_key)
    prompt = (
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
    return jsonify({"status": "Song Reactor Bot is live 🎵", "usage": "POST /react with field 'audio' (MP3 file)"})


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

    try:
        lyrics = transcribe(tmp_path)
        bpm = detect_bpm(tmp_path)
        reaction = get_groq_reaction(lyrics, bpm, api_key)
        return jsonify({"bpm": bpm, "lyrics": lyrics, "reaction": reaction})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.unlink(tmp_path)  # Always clean up the temp file


# ---------------------------------------------------------------------------
# CLI entry point (local use only)
# ---------------------------------------------------------------------------

def main():
    api_key = _get_api_key()
    if api_key == "YOUR_GROQ_API_KEY":
        print("Error: Set GROQ_API_KEY env var or edit reactor.py.")
        print("Free key at https://console.groq.com")
        sys.exit(1)

    audio_path = sys.argv[1] if len(sys.argv) > 1 else AUDIO_FILE
    if not os.path.isfile(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        print("Usage: python reactor.py path/to/song.mp3")
        sys.exit(1)

    lyrics = transcribe(audio_path)
    bpm = detect_bpm(audio_path)
    reaction = get_groq_reaction(lyrics, bpm, api_key)

    print("=" * 60)
    print("TIKTOK LIVE REACTION 🎵")
    print("=" * 60)
    print(reaction)
    print("=" * 60)


if __name__ == "__main__":
    # Running directly → CLI mode
    # Running via gunicorn → gunicorn imports `app` directly (main() is never called)
    main()
