import os
import json
import re
import random
import logging
import threading
import subprocess
import zipfile
import requests
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv
from mutagen.flac import FLAC, Picture

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

load_dotenv(Path(os.environ.get("ALBUMGEN_ENV_FILE", BASE_DIR / ".env")))


def _cfg(name, default=None):
    return os.environ.get(name, default)


LLM_URL = _cfg("LLM_URL", "http://10.0.1.27:8080/")
MUSICGEN_DIR = Path(_cfg("MUSICGEN_DIR", "/opt/musicgen")).resolve()
SDGUI_DIR = Path(_cfg("SDGUI_DIR", "/opt/sd-gui")).resolve()

SYNTH_MODEL = _cfg("SYNTH_MODEL", "acestep-v15-xl-turbo-Q8_0.gguf")
LM_MODEL = _cfg("LM_MODEL", "acestep-5Hz-lm-4B-Q8_0.gguf")
INFERENCE_STEPS = int(_cfg("INFERENCE_STEPS", "10"))
COVER_W = int(_cfg("COVER_WIDTH", "768"))
COVER_H = int(_cfg("COVER_HEIGHT", "768"))
COVER_STEPS = int(_cfg("COVER_STEPS", "10"))
COVER_MODEL = _cfg("COVER_MODEL", "z_image_turbo-Q8_0.gguf")
COVER_LLM = _cfg("COVER_LLM", "qwen_3_4b-Q8_0.gguf")
COVER_VAE = _cfg("COVER_VAE", "ae.safetensors")
DURATION_JITTER = float(_cfg("DURATION_JITTER", "0.08"))
TARGET_DURATION_DEFAULT = int(_cfg("TARGET_DURATION_DEFAULT", "180"))

ACE_LM = MUSICGEN_DIR / "bin" / "ace-lm"
ACE_SYNTH = MUSICGEN_DIR / "bin" / "ace-synth"
ACE_MODELS = MUSICGEN_DIR / "models"
SD_CLI = SDGUI_DIR / "bin" / "sd-cli"
SD_MODELS = SDGUI_DIR / "models"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

TRACK_DEFAULTS = {
    "caption": "", "lyrics": "[instrumental]", "bpm": 0, "duration": 0, "keyscale": "",
    "timesignature": "", "vocal_language": "en", "seed": 0, "lm_batch_size": 1,
    "synth_batch_size": 1, "lm_temperature": 0.5, "lm_cfg_scale": 7,
    "lm_top_p": 0.5, "lm_top_k": 0,
    "lm_negative_prompt": "bad audio, robotic vocals, autotune, distortion, spoken word, overly loud backing vocals, midi artifact, mechanical piano, glitchy drums, overcompressed, muddy mix, muddy bass, heavy reverb, crowd noise, background noise, unwanted silence, chaotic arrangement, predictable loops, repetitive",
    "negative_prompt": "bad audio, robotic vocals, autotune, distortion, spoken word, overly loud backing vocals, midi artifact, mechanical piano, glitchy drums, overcompressed, muddy mix, muddy bass, heavy reverb, crowd noise, background noise, unwanted silence, chaotic arrangement, predictable loops, repetitive",
    "use_cot_caption": True,
    "audio_codes": "", "inference_steps": INFERENCE_STEPS, "guidance_scale": 0.0, "shift": 10,
    "dcw_scaler": 0.0, "dcw_high_scaler": 0.0, "dcw_mode": "low",
    "audio_cover_strength": 1.0, "cover_noise_strength": 0.0, "repainting_start": 0,
    "repainting_end": -1, "latent_shift": 0.0, "latent_rescale": 1.0,
    "custom_timesteps": "", "task_type": "text2music", "track": "", "solver": "euler",
    "lm_mode": "generate", "output_format": "wav32", "peak_clip": 10, "mp3_bitrate": 128,
    "synth_model": SYNTH_MODEL, "lm_model": LM_MODEL,
    "adapter": "", "adapter_scale": 1.0
}

INT_FIELDS = {
    "bpm", "duration", "seed", "lm_batch_size", "synth_batch_size",
    "inference_steps", "repainting_start", "repainting_end",
    "peak_clip", "mp3_bitrate"
}
FLOAT_FIELDS = {
    "lm_temperature", "lm_cfg_scale", "lm_top_p", "lm_top_k",
    "guidance_scale", "shift", "dcw_scaler", "dcw_high_scaler",
    "audio_cover_strength", "cover_noise_strength", "latent_shift",
    "latent_rescale", "adapter_scale"
}


def safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def coerce_track_types(track: dict) -> dict:
    coerced = {}
    for k, v in track.items():
        if k in INT_FIELDS:
            coerced[k] = safe_int(v, TRACK_DEFAULTS.get(k, 0))
        elif k in FLOAT_FIELDS:
            coerced[k] = safe_float(v, TRACK_DEFAULTS.get(k, 0.0))
        else:
            coerced[k] = v
    return coerced


def slugify(text, fallback="untitled"):
    s = re.sub(r"[^\w\s-]", "", str(text), flags=re.UNICODE).strip()
    s = re.sub(r"[\s\-]+", " ", s)
    s = re.sub(r" +", " ", s).strip()
    return s[:64] or fallback


def extract_json_from_llm_response(content: str):
    if not content or not content.strip():
        raise ValueError("Empty LLM response content")

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    patterns = [
        r'```json\s*\n([\s\S]*?)\n```',
        r'```\s*\n([\s\S]*?)\n```',
        r'```json([\s\S]*?)```',
        r'```([\s\S]*?)```',
        r'\{[\s\S]*\}',
        r'\[[\s\S]*\]',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            candidate = match.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    continue

    first_brace = content.find('{')
    first_bracket = content.find('[')

    if first_brace != -1:
        last_brace = content.rfind('}')
        if last_brace > first_brace:
            candidate = content[first_brace:last_brace + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(re.sub(r',\s*}', '}', candidate))
                except json.JSONDecodeError:
                    pass

    if first_bracket != -1:
        last_bracket = content.rfind(']')
        if last_bracket > first_bracket:
            candidate = content[first_bracket:last_bracket + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(re.sub(r',\s*]', ']', candidate))
                except json.JSONDecodeError:
                    pass

    raise ValueError("Could not parse JSON from LLM response")


def strip_code_fences(content: str) -> str:
    text = content.strip()
    fence = re.match(r'^```[a-zA-Z]*\s*\n([\s\S]*?)\n?```$', text)
    if fence:
        return fence.group(1).strip()
    return text


def call_llm(url: str, messages: list, temperature: float, max_tokens: int,
             disable_thinking: bool = False) -> str:
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    resp = requests.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=6000
    )
    if resp.status_code == 400 and disable_thinking:
        logger.info("Server rejected chat_template_kwargs; retrying without it.")
        payload.pop("chat_template_kwargs", None)
        resp = requests.post(
            f"{url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=600
        )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    if not content.strip():
        reasoning = message.get("reasoning_content") or ""
        logger.warning(
            "LLM returned empty content. finish_reason=%r message_keys=%s reasoning_content_present=%s "
            "reasoning_start=%r",
            choice.get("finish_reason"), sorted(message.keys()), bool(reasoning), reasoning[:200]
        )
    return content


def clean_text_answer(content: str) -> str:
    """Extract a clean single-line answer (name/prompt) from an LLM response."""
    text = strip_code_fences(content or "")
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[*_`#>]+", "", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    line = re.sub(r'^(album\s*name|song\s*title|title|name|prompt)\s*:\s*', '',
                  lines[0], flags=re.IGNORECASE)
    return line.strip().strip('"').strip("'").strip().strip('"').strip()


def build_tracks_system_prompt(guide_content: str) -> str:
    return (
        "You are an expert AI music producer, lyricist, and prompt engineer for ACE-Step 1.5.\n\n"
        f"{guide_content}\n\n"
        "## CRITICAL INSTRUCTION FOR AUDIO SYNTHESIS\n"
        "EVERY LINE outside of brackets WILL BE SUNG AS LYRICS. There is no narration, "
        "description, or stage direction between tags — only words that will be vocalized.\n\n"
        "Never use parentheses () for musical or production descriptions. All structure tags, instrumental "
        "breaks, and musical cues MUST be delimited strictly with brackets []. If a section has no "
        "vocals, use a single bracketed tag with no lines following it "
        "(e.g. [Intro Instrumental], [Guitar Solo], [Musical Interlude], [Outro Instrumental]).\n"
        "If a section has vocals, use the [Tag - modifier] pattern to keep style cues inside "
        "the brackets (e.g. [Outro -spoken words, fading out], [Chorus -anthemic]). The lines after "
        "the tag must be actual singable lyrics.\n\n"
        "## Your Task\n"
        "The user will provide a style description, a number of tracks, and a target duration. "
        "You are creating ONE COHESIVE ALBUM: every track must feel like it belongs on the same record, "
        "yet each track must be a distinct, interesting song with its own identity and angle on the style.\n\n"
        "Return ONLY valid JSON: a single track configuration object, or a JSON array of such "
        "objects.\n\n"
        "## Titles (CRITICAL)\n"
        "- Every track object MUST include a 'title' key: a unique, evocative SONG NAME (not a description).\n"
        "- No two tracks may share a title. No numbering, no quotes, no colons. Max ~50 characters.\n"
        "- Titles must fit the album's identity.\n\n"
        "## Durations (CRITICAL)\n"
        "- The user gives a TARGET duration in seconds. Each track's 'duration' must be NEAR that target: "
        "within about +/-10%, never identical across all tracks (small natural variation is desired).\n"
        "- Match lyric length to each track's duration:\n"
        "  * <60s: 1 verse + 1 chorus (6-10 lines total)\n"
        "  * 60-120s: 1-2 verses + 2 choruses\n"
        "  * 120-180s: 2 verses + 2 choruses + optional bridge\n"
        "  * >180s: 2-3 verses + 2-3 choruses + bridge + intro/outro\n"
        "- Each lyric line should be 6-10 syllables. Use blank lines between sections.\n\n"
        "## Lyrics Requirements (CRITICAL)\n"
        "- If the song is instrumental, set lyrics to '[instrumental]'.\n"
        "- Lyrics MUST include proper song structure tags: [Intro], [Verse 1], [Verse 2], [Chorus], "
        "[Bridge], [Guitar Solo], [Keyboard Interlude], [Outro Instrumental], etc.\n"
        "- Use brackets [] for ALL structural and musical cues. Never use parentheses ().\n\n"
        "## Caption-Lyrics Consistency\n"
        "- Instruments in Caption must match Instrumental section tags in Lyrics.\n"
        "- Emotion in Caption must match Energy tags in Lyrics.\n"
        "- Vocal description in Caption must match Vocal control tags in Lyrics.\n\n"
        "## Avoiding AI-Flavored Lyrics\n"
        "- No adjective stacking, no inconsistent rhyme patterns, no blurred section boundaries.\n"
        "- Keep lines singable (6-10 syllables). Stick to one core metaphor per song.\n\n"
        "## Album-Wide Uniqueness (CRITICAL - DO NOT REPEAT)\n"
        "- Write EVERY track from scratch. Never reuse a line, couplet, verse, bridge, or chorus "
        "from another track, even when the songs share a mood or theme.\n"
        "- The same line must never appear in more than one track. Once you have written a verse or "
        "chorus, that exact wording is spent and must not be copied into any other track.\n"
        "- Each track needs a distinct chorus hook (its own title line, imagery, and closing couplet). "
        "Do not fall back on stock filler phrases such as 'in the silence we remain', "
        "'when time takes its toll', 'we still have each other', or 'find our way home'.\n"
        "- Within a single track, Verse 1 and Verse 2 must differ; never repeat a verse.\n"
        "- Vary scenery and object imagery between tracks (streets, windows, rain, trains, stars, "
        "candles, records, coffee) so no two songs conjure the same picture.\n"
        "- Give each track its own narrative angle and central image while keeping the album cohesive.\n"
        "- Before finalizing, re-read all tracks written so far; if any line echoes an earlier one, "
        "rewrite it.\n\n"
        "## Track Object Keys\n"
        "Each track must have:\n"
        "- title: Unique short song name (see Titles rules above).\n"
        "- caption: A dense descriptive paragraph detailing genre, instruments, production style, vocals, "
        "and mood. Do NOT include BPM/key/tempo here.\n"
        "- lyrics: Full structured lyrics with bracketed tags, or '[instrumental]' for instrumental tracks.\n"
        "- bpm: Number (30-300) or 0 to auto-infer.\n"
        "- duration: Number in seconds, near the target duration.\n"
        "- keyscale: e.g. 'C Major', 'Am', or empty string to auto-infer.\n"
        "- timesignature: e.g. '4/4', '3/4', or empty string.\n"
        "- vocal_language: Language code (e.g. 'en') or empty string.\n"
        "- seed: 0 for random.\n"
        "- lm_temperature: Float 0.1-1.5, default 0.5.\n"
        "- lm_cfg_scale: Float, default 7.\n"
        "- lm_top_p: Float, default 0.5.\n"
        "- lm_top_k: Int, default 0.\n"
        "- shift: Float, default 10.\n\n"
        "Vary tempos, moods, and energy across tracks while staying true to the album's overall style.\n\n"
        "Return ONLY valid JSON — no explanations, no markdown, no code fences."
    )


def build_reference_system_prompt(guide_content: str) -> str:
    return (
        "You are an expert AI music producer, lyricist, and prompt engineer for ACE-Step 1.5, "
        "specializing in reference-guided generation.\n\n"
        f"{guide_content}\n\n"
        "## REFERENCE MODE (CRITICAL)\n"
        "This album will be generated using an uploaded reference music track. Every track draws on "
        "that reference audio for timbre and character through a timbre reference, while the "
        "audio_cover_strength value controls how closely each track follows the generated arrangement "
        "(higher strength stays closer to the planned structure, lower strength invents more freely).\n\n"
        "Adapt each track to the reference:\n"
        "- The caption must describe the STYLE you intend to apply (the genre, instruments, production, "
        "and mood the user wants), informed by the reference track's character, not a generic "
        "description.\n"
        "- Keep the album cohesive: every track should feel like a distinct interpretation sharing the "
        "reference's character, with its own angle on the style.\n\n"
        "## CRITICAL INSTRUCTION FOR AUDIO SYNTHESIS\n"
        "EVERY LINE outside of brackets WILL BE SUNG AS LYRICS. There is no narration, "
        "description, or stage direction between tags — only words that will be vocalized.\n\n"
        "Never use parentheses () for musical or production descriptions. All structure tags, instrumental "
        "breaks, and musical cues MUST be delimited strictly with brackets []. If a section has no "
        "vocals, use a single bracketed tag with no lines following it "
        "(e.g. [Intro Instrumental], [Guitar Solo], [Musical Interlude], [Outro Instrumental]).\n"
        "If a section has vocals, use the [Tag - modifier] pattern to keep style cues inside "
        "the brackets (e.g. [Outro -spoken words, fading out], [Chorus -anthemic]). The lines after "
        "the tag must be actual singable lyrics.\n\n"
        "## Your Task\n"
        "The user will provide a style description, a number of tracks, and a target duration. "
        "You are creating ONE COHESIVE ALBUM: every track must feel like it belongs on the same record, "
        "yet each track must be a distinct, interesting reinterpretation of the cover reference with "
        "its own identity and angle on the style.\n\n"
        "Return ONLY valid JSON: a single track configuration object, or a JSON array of such "
        "objects.\n\n"
        "## Titles (CRITICAL)\n"
        "- Every track object MUST include a 'title' key: a unique, evocative SONG NAME (not a description).\n"
        "- No two tracks may share a title. No numbering, no quotes, no colons. Max ~50 characters.\n"
        "- Titles must fit the album's identity.\n\n"
        "## Durations (CRITICAL)\n"
        "- The user gives a TARGET duration in seconds. Each track's 'duration' must be NEAR that target: "
        "within about +/-10%, never identical across all tracks (small natural variation is desired).\n"
        "- Match lyric length to each track's duration:\n"
        "  * <60s: 1 verse + 1 chorus (6-10 lines total)\n"
        "  * 60-120s: 1-2 verses + 2 choruses\n"
        "  * 120-180s: 2 verses + 2 choruses + optional bridge\n"
        "  * >180s: 2-3 verses + 2-3 choruses + bridge + intro/outro\n"
        "- Each lyric line should be 6-10 syllables. Use blank lines between sections.\n\n"
        "## Lyrics Requirements (CRITICAL)\n"
        "- If the song is instrumental, set lyrics to '[instrumental]'.\n"
        "- Lyrics MUST include proper song structure tags: [Intro], [Verse 1], [Verse 2], [Chorus], "
        "[Bridge], [Guitar Solo], [Keyboard Interlude], [Outro Instrumental], etc.\n"
        "- Use brackets [] for ALL structural and musical cues. Never use parentheses ().\n\n"
        "## Caption-Lyrics Consistency\n"
        "- Instruments in Caption must match Instrumental section tags in Lyrics.\n"
        "- Emotion in Caption must match Energy tags in Lyrics.\n"
        "- Vocal description in Caption must match Vocal control tags in Lyrics.\n\n"
        "## Avoiding AI-Flavored Lyrics\n"
        "- No adjective stacking, no inconsistent rhyme patterns, no blurred section boundaries.\n"
        "- Keep lines singable (6-10 syllables). Stick to one core metaphor per song.\n\n"
        "## Album-Wide Uniqueness (CRITICAL - DO NOT REPEAT)\n"
        "- Write EVERY track from scratch. Never reuse a line, couplet, verse, bridge, or chorus "
        "from another track, even when the songs share a mood or theme.\n"
        "- The same line must never appear in more than one track. Once you have written a verse or "
        "chorus, that exact wording is spent and must not be copied into any other track.\n"
        "- Each track needs a distinct chorus hook (its own title line, imagery, and closing couplet). "
        "Do not fall back on stock filler phrases such as 'in the silence we remain', "
        "'when time takes its toll', 'we still have each other', or 'find our way home'.\n"
        "- Within a single track, Verse 1 and Verse 2 must differ; never repeat a verse.\n"
        "- Vary scenery and object imagery between tracks (streets, windows, rain, trains, stars, "
        "candles, records, coffee) so no two songs conjure the same picture.\n"
        "- Give each track its own narrative angle and central image while keeping the album cohesive.\n"
        "- Before finalizing, re-read all tracks written so far; if any line echoes an earlier one, "
        "rewrite it.\n\n"
        "## Track Object Keys\n"
        "Each track must have:\n"
        "- title: Unique short song name (see Titles rules above).\n"
        "- caption: A dense descriptive paragraph detailing the track's genre, instruments, production "
        "style, vocals, and mood. Do NOT include BPM/key/tempo here.\n"
        "- lyrics: Full structured lyrics with bracketed tags, or '[instrumental]' for instrumental tracks.\n"
        "- bpm: Number (30-300) or 0 to auto-infer.\n"
        "- duration: Number in seconds, near the target duration.\n"
        "- keyscale: e.g. 'C Major', 'Am', or empty string to auto-infer.\n"
        "- timesignature: e.g. '4/4', '3/4', or empty string.\n"
        "- vocal_language: Language code (e.g. 'en') or empty string.\n"
        "- seed: 0 for random.\n"
        "- lm_temperature: Float 0.1-1.5, default 0.5.\n"
        "- lm_cfg_scale: Float, default 7.\n"
        "- lm_top_p: Float, default 0.5.\n"
        "- lm_top_k: Int, default 0.\n"
        "- shift: Float, default 10.\n\n"
        "Vary tempos, moods, and energy across tracks while staying true to the album's overall style "
        "and the reference track.\n\n"
        "Return ONLY valid JSON — no explanations, no markdown, no code fences."
    )


class AlbumGeneratorApp:
    def __init__(self):
        self.app = Flask(__name__, template_folder='templates')
        self.app.config['SECRET_KEY'] = 'albumgenerator_secret_key_12345'

        guide_path = BASE_DIR / "Song Writing Guide.md"
        if guide_path.exists():
            self.song_writing_guide = guide_path.read_text(encoding='utf-8')
            logger.info("Song Writing Guide loaded successfully.")
        else:
            self.song_writing_guide = ""
            logger.warning("Song Writing Guide.md not found.")

        self.job_lock = threading.Lock()
        self.job_reset()

        self._setup_routes()

    def job_reset(self):
        self.job = {
            "running": False, "stage": "idle", "message": "", "detail": "",
            "percent": 0, "album_name": None, "zip_name": None,
            "error": None, "done": False,
            "cover_ready": False, "tracks": []
        }

    def job_update(self, **kwargs):
        with self.job_lock:
            self.job.update(kwargs)

    def job_track_update(self, index, status, error=None, audio_url=None):
        with self.job_lock:
            if 0 <= index < len(self.job["tracks"]):
                self.job["tracks"][index]["status"] = status
                if error is not None:
                    self.job["tracks"][index]["error"] = error
                if audio_url is not None:
                    self.job["tracks"][index]["audio_url"] = audio_url

    def _clean_output_dir(self):
        for f in OUTPUT_DIR.iterdir():
            try:
                if f.is_file() or f.is_symlink():
                    f.unlink()
                elif f.is_dir():
                    import shutil
                    shutil.rmtree(f)
            except Exception as e:
                logger.warning(f"Could not delete {f}: {e}")

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template(
                'index.html',
                defaults=TRACK_DEFAULTS,
                target_duration=TARGET_DURATION_DEFAULT,
                cover_size=f"{COVER_W}x{COVER_H}",
                llm_url=LLM_URL.rstrip('/')
            )

        @self.app.route('/start', methods=['POST'])
        def start():
            data = request.json or {}
            style = (data.get('style') or '').strip()
            num_tracks = safe_int(data.get('num_tracks', 1), 1)
            target_duration = safe_int(data.get('target_duration', TARGET_DURATION_DEFAULT),
                                       TARGET_DURATION_DEFAULT)
            llm_url = (data.get('llm_url') or LLM_URL).strip().rstrip('/')

            if not style:
                return jsonify({"status": "error", "message": "Style description is required."}), 400
            if num_tracks < 1 or num_tracks > 50:
                return jsonify({"status": "error", "message": "Track count must be between 1 and 50."}), 400
            if target_duration < 10 or target_duration > 1200:
                return jsonify({"status": "error", "message": "Target duration must be between 10 and 1200 seconds."}), 400

            prev_zip = None
            with self.job_lock:
                if self.job["running"]:
                    return jsonify({"status": "error", "message": "A generation job is already running."}), 409
                prev_zip = self.job.get("zip_name")
                self.job_reset()
                self.job["running"] = True
                self.job["stage"] = "queued"
                self.job["message"] = "Queued..."

            if prev_zip:
                try:
                    (BASE_DIR / prev_zip).unlink(missing_ok=True)
                    logger.info(f"Removed previous album zip on new run: {prev_zip}")
                except Exception as e:
                    logger.warning(f"Could not remove previous zip {prev_zip}: {e}")

            params = {
                "style": style, "num_tracks": num_tracks,
                "target_duration": target_duration, "llm_url": llm_url,
                "ref_audio_filename": (data.get('ref_audio_filename') or '').strip(),
                "audio_cover_strength": safe_float(data.get('audio_cover_strength', 1.0), 1.0)
            }
            t = threading.Thread(target=self._run_pipeline, args=(params,), daemon=True)
            t.start()
            return jsonify({"status": "started"})

        @self.app.route('/status')
        def status():
            with self.job_lock:
                return jsonify(dict(self.job))

        @self.app.route('/preview/cover')
        def preview_cover():
            cover = OUTPUT_DIR / "cover.jpg"
            if cover.exists():
                return send_file(cover, mimetype='image/jpeg')
            return jsonify({"error": "Cover not available yet"}), 404

        @self.app.route('/preview/audio/<path:filename>')
        def preview_audio(filename):
            allowed = set()
            for f in OUTPUT_DIR.iterdir():
                if f.is_file() and f.suffix.lower() == '.flac':
                    allowed.add(f.name)
                elif f.is_file() and f.suffix.lower() == '.wav':
                    allowed.add(f.name)
            if filename not in allowed:
                return jsonify({"error": "File not found"}), 404
            audio_path = OUTPUT_DIR / filename
            if filename.endswith('.flac'):
                return send_file(audio_path, mimetype='audio/flac')
            return send_file(audio_path, mimetype='audio/wav')

        @self.app.route('/download')
        def download():
            with self.job_lock:
                zip_name = self.job.get("zip_name")
            if not zip_name:
                return jsonify({"status": "error", "message": "No album ready for download."}), 404
            zip_path = BASE_DIR / zip_name
            if not zip_path.exists():
                return jsonify({"status": "error", "message": "Zip file not found."}), 404
            logger.info(f"Serving album zip for download: {zip_name}")
            return send_file(zip_path, as_attachment=True, download_name=zip_name)

        @self.app.route('/upload/ref_audio', methods=['POST'])
        def upload_ref_audio():
            if 'ref_audio' not in request.files or not request.files['ref_audio'].filename:
                return jsonify({"status": "error", "message": "No audio file provided."}), 400
            for f in UPLOADS_DIR.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            file = request.files['ref_audio']
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_name = f"ref_{timestamp}_{filename}"
            file_path = UPLOADS_DIR / saved_name
            file.save(file_path)
            logger.info(f"Saved reference audio: {saved_name}")
            return jsonify({"status": "ok", "filename": saved_name})

        @self.app.route('/clear/ref_audio', methods=['POST'])
        def clear_ref_audio():
            for f in UPLOADS_DIR.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            return jsonify({"status": "ok"})

    def _run_pipeline(self, params):
        style = params["style"]
        num_tracks = params["num_tracks"]
        target_duration = params["target_duration"]
        llm_url = params["llm_url"]
        ref_audio_filename = params.get("ref_audio_filename", "")
        audio_cover_strength = params.get("audio_cover_strength", 1.0)

        ref_audio_path = None
        if ref_audio_filename:
            candidate = UPLOADS_DIR / ref_audio_filename
            if candidate.exists():
                ref_audio_path = candidate
                logger.info(f"Reference audio found: {ref_audio_path.name}")

        try:
            self._clean_output_dir()

            with open(OUTPUT_DIR / "request.txt", 'w', encoding='utf-8') as f:
                f.write("Original Request\n")
                f.write("================\n\n")
                f.write(f"Style: {style}\n")
                f.write(f"Number of tracks: {num_tracks}\n")
                f.write(f"Target duration (seconds): {target_duration}\n")
                if ref_audio_path:
                    f.write(f"Reference: {ref_audio_path.name}\n")
                    f.write(f"Reference strength: {audio_cover_strength}\n")

            dur_min = max(10, int(target_duration * (1 - DURATION_JITTER)))
            dur_max = int(target_duration * (1 + DURATION_JITTER))

            self.job_update(stage="planning", message="Writing songs...",
                            detail="Asking the LLM to plan your album...", percent=2)

            tracks = self._plan_tracks(llm_url, style, num_tracks, target_duration,
                                       ref_audio_path=ref_audio_path)

            self.job_update(album_name=tracks["album_name"],
                            message="Album planned", detail=tracks["album_name"], percent=6)

            used_slugs = set()
            track_entries = []
            for i, raw in enumerate(tracks["tracks"]):
                merged = {**TRACK_DEFAULTS, **raw}
                merged.pop("title", None)
                merged = coerce_track_types(merged)
                merged["seed"] = 0
                merged["inference_steps"] = INFERENCE_STEPS
                merged["synth_model"] = SYNTH_MODEL
                merged["lm_model"] = LM_MODEL
                if ref_audio_path:
                    merged["task_type"] = "text2music"
                    merged["audio_cover_strength"] = audio_cover_strength

                d = safe_int(merged.get("duration"), 0)
                if d < dur_min or d > dur_max:
                    d = random.randint(dur_min, dur_max)
                merged["duration"] = d

                slug = slugify(raw.get("title") or f"Track {i+1}", f"Track {i+1}")
                base_slug = slug
                n = 2
                while slug in used_slugs:
                    slug = f"{base_slug} {n}"
                    n += 1
                used_slugs.add(slug)
                track_entries.append({"title": raw.get("title") or f"Track {i+1}", "slug": slug,
                                      "json": OUTPUT_DIR / f"{i+1:02d} - {slug}.json"})
                with open(OUTPUT_DIR / f"{i+1:02d} - {slug}.json", 'w', encoding='utf-8') as f:
                    json.dump(merged, f, indent=4)

            self.job_update(tracks=[{"title": t["title"], "slug": t["slug"], "status": "pending",
                                     "audio_url": None} for t in track_entries])

            wav_paths = []
            span_per_track = 84.0 / len(track_entries)
            for i, entry in enumerate(track_entries):
                pct = 6 + int(i * span_per_track)
                self.job_update(stage="music",
                                message=f"Generating music ({i+1}/{len(track_entries)})",
                                detail=f"Track {i+1}: \"{entry['title']}\"", percent=pct)
                self.job_track_update(i, "working")
                try:
                    wav_path = self._generate_track(entry, i+1, ref_audio_path=ref_audio_path)
                    track_num = i + 1
                    flac_name = f"{track_num:02d} - {entry['slug']}.flac"
                    flac_path = OUTPUT_DIR / flac_name
                    audio_url = f"/preview/audio/{flac_name}" if flac_path.exists() else f"/preview/audio/{wav_path.name}"
                    wav_paths.append(wav_path)
                    self.job_track_update(i, "done", audio_url=audio_url)
                    logger.info(f"Track {i+1}/{len(track_entries)} done: {entry['title']}")
                except Exception as e:
                    logger.exception(f"Track {i+1} failed: {entry['title']}")
                    self.job_track_update(i, "failed", error=str(e))
                    self.job_update(detail=f"Track failed: {e}")

            self.job_update(stage="cover", message="Now generating cover art...",
                            detail="Rendering album cover with Stable Diffusion...", percent=92)
            self._generate_cover(tracks["cover_prompt"])
            self.job_update(cover_ready=True)

            self.job_update(stage="bundling", message="Bundling album...",
                            detail="Creating zip archive...", percent=97)
            zip_name = slugify(tracks["album_name"], "album") + ".zip"
            self._bundle_zip(zip_name, track_entries, wav_paths)

            self.job_update(stage="done", running=False, done=True, zip_name=zip_name,
                            message="Album complete!", detail="Ready to download.",
                            percent=100)
            logger.info(f"Album complete: {tracks['album_name']} ({zip_name})")

        except Exception as e:
            logger.exception("Album generation failed")
            self.job_update(running=False, done=True, stage="error", error=str(e),
                            message="Generation failed.", detail=str(e), percent=100)

    @staticmethod
    def _lyric_lines(lyrics):
        lines = []
        for ln in (lyrics or "").splitlines():
            s = ln.strip()
            if not s or (s.startswith("[") and s.endswith("]")):
                continue
            lines.append(re.sub(r"\s+", " ", s).lower())
        return lines

    def _repeated_lines(self, candidate_lyrics, prev_tracks):
        candidate = set(self._lyric_lines(candidate_lyrics))
        if not candidate:
            return set()
        previous = set()
        for pt in prev_tracks:
            previous.update(self._lyric_lines(pt.get("lyrics")))
        return candidate & previous

    def _prior_tracks_context(self, prev_tracks, max_chars=48000):
        if not prev_tracks:
            return ""
        titles = "; ".join(f"{j}. {pt.get('title', '')}" for j, pt in enumerate(prev_tracks, 1))
        blocks = []
        for j, pt in enumerate(prev_tracks, 1):
            body = (pt.get("lyrics") or "").strip()
            head = (f"### Track {j}: {pt.get('title', '')} "
                    f"(duration {pt.get('duration', 0)}s, bpm {pt.get('bpm', 0)})")
            blocks.append(f"{head}\n{body}")
        joined = "\n\n".join(blocks)
        if len(joined) > max_chars:
            kept, total = [], 0
            for block in reversed(blocks):
                if kept and total + len(block) > max_chars:
                    break
                kept.append(block)
                total += len(block)
            joined = "\n\n".join(reversed(kept))
        return f"Titles already used (all forbidden): {titles}\n\n{joined}"

    def _plan_single_track(self, llm_url, system_prompt, style, target_duration,
                           index, total, prev_tracks, avoid_lines=None):
        context = self._prior_tracks_context(prev_tracks)
        prior_section = ""
        if context:
            prior_section = (
                "\n\n## TRACKS ALREADY WRITTEN - ALL OF THIS IS OFF-LIMITS\n"
                "The songs below are already on the album. Your new track MUST NOT reuse, copy, or "
                "closely paraphrase any line, image, rhyme, chorus hook, verse, or bridge from them. "
                "Study them, then deliberately write something different.\n\n"
                f"{context}\n"
            )
        avoid_section = ""
        if avoid_lines:
            shown = " | ".join(sorted(avoid_lines)[:12])
            avoid_section = (
                f"\n\nYou previously reused these exact lines, which is forbidden: {shown}. "
                "Do not use them or any close variation; write entirely new lines.\n"
            )
        user_prompt = (
            f"Album style:\n{style}\n\n"
            f"Target duration per track: approximately {target_duration} seconds "
            f"(this track should land within +/-10% of that).\n\n"
            f"Write track {index} of {total}.\n"
            f"{prior_section}{avoid_section}\n"
            f"Return a JSON array containing exactly ONE track object for track {index} "
            f"(its title must not duplicate any title listed above)."
        )
        logger.info(f"Calling LLM at {llm_url}/v1/chat/completions for track {index}/{total}...")
        content = call_llm(llm_url, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], 0.8, 8192)
        parsed = extract_json_from_llm_response(content)
        if isinstance(parsed, dict):
            track = parsed
        elif isinstance(parsed, list) and parsed:
            track = parsed[0]
        else:
            raise ValueError(f"Track {index} did not return a valid object.")
        if not isinstance(track, dict):
            raise ValueError(f"Track {index} is not a valid object.")
        if not track.get("title"):
            track["title"] = f"Track {index}"
        return track

    def _plan_tracks(self, llm_url, style, num_tracks, target_duration, ref_audio_path=None):
        system_prompt = (build_reference_system_prompt(self.song_writing_guide) if ref_audio_path
                         else build_tracks_system_prompt(self.song_writing_guide))
        parsed = []
        for i in range(num_tracks):
            self.job_update(message=f"Writing song {i+1}/{num_tracks}...",
                            detail=f"Planning track {i+1} of {num_tracks}...",
                            percent=2 + int((i / max(num_tracks, 1)) * 3))
            track = None
            avoid_lines = None
            for attempt in range(3):
                try:
                    candidate = self._plan_single_track(
                        llm_url, system_prompt, style, target_duration,
                        i + 1, num_tracks, parsed, avoid_lines=avoid_lines)
                except Exception as e:
                    logger.warning(f"Track {i+1} attempt {attempt + 1} failed to plan: {e}")
                    continue
                dups = self._repeated_lines(candidate.get("lyrics"), parsed)
                if not dups:
                    track = candidate
                    break
                logger.warning(
                    f"Track {i+1} attempt {attempt + 1} reused {len(dups)} line(s) "
                    f"from earlier tracks: {sorted(dups)[:3]}"
                )
                track = candidate
                avoid_lines = dups
            if track is None:
                raise ValueError(f"Failed to plan track {i + 1} after multiple attempts.")
            parsed.append(track)
            logger.info(f"Planned track {i+1}/{num_tracks}: {track.get('title')}")

        album_name = ""
        raw_snippet = ""
        for attempt in range(2):
            try:
                logger.info(f"Calling LLM for album name (attempt {attempt + 1}/2)...")
                content = call_llm(llm_url, [
                    {"role": "system", "content": (
                        "You are a creative music industry naming expert. Given a music style description, "
                        "invent ONE compelling album name that fits the style and mood. "
                        "Return ONLY the album name as plain text — no quotes, no explanations, no markdown."
                    )},
                    {"role": "user", "content": f"Music style description:\n\n{style}"}
                ], 0.9 if attempt == 0 else 0.7, 4096, disable_thinking=True)
                raw_snippet = (content or "")[:200]
                candidate = clean_text_answer(content)
                if candidate and len(candidate) >= 2 and re.search(r"[A-Za-z0-9]", candidate):
                    album_name = candidate[:80]
                    break
                logger.warning(f"Album name attempt {attempt + 1} unusable. Raw response start: {raw_snippet!r}")
            except Exception as e:
                logger.warning(f"Album name attempt {attempt + 1} failed: {e}")
        if not album_name:
            logger.warning(
                f"Falling back to 'Untitled Album'. Last raw LLM response (first 200 chars): {raw_snippet!r}"
            )
            album_name = "Untitled Album"

        cover_prompt = ""
        try:
            logger.info("Calling LLM for album cover prompt...")
            content = call_llm(llm_url, [
                {"role": "system", "content": (
                    "You are an expert at writing Stable Diffusion image generation prompts. "
                    "Given a music style description, write ONE detailed album cover art prompt. "
                    "Describe subject, style, mood, color palette, composition, lighting, and artistic medium. "
                    "The image will be square (768x768), suitable as album cover art. "
                    "Return ONLY the prompt as plain text — no explanations, no quotes, no markdown."
                )},
                {"role": "user", "content": f"Music style description:\n\n{style}\n\nAlbum name: {album_name}"}
            ], 0.9, 4096, disable_thinking=True)
            cover_prompt = clean_text_answer(content)[:1500]
            if not cover_prompt:
                logger.warning(f"Cover prompt response unusable. Raw response start: {(content or '')[:200]!r}")
        except Exception as e:
            logger.warning(f"Cover prompt generation failed: {e}")
        if not cover_prompt:
            raise ValueError("Failed to generate an album cover prompt.")

        with open(OUTPUT_DIR / "album_name.txt", 'w', encoding='utf-8') as f:
            f.write(album_name + "\n")
        with open(OUTPUT_DIR / "album_cover.txt", 'w', encoding='utf-8') as f:
            f.write(cover_prompt + "\n")

        return {"album_name": album_name, "cover_prompt": cover_prompt, "tracks": parsed}

    def _run_cmd(self, cmd, timeout):
        logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
        result = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(OUTPUT_DIR),
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            detail = f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
            raise RuntimeError(detail)
        return result

    def _generate_track(self, entry, track_num, ref_audio_path=None):
        json_path = entry["json"]
        stem = json_path.stem

        llm_json = json_path.with_name(f"{stem}0.json")
        llm_json.unlink(missing_ok=True)

        try:
            logger.info(f"Running internal LLM enhancement for: {entry['title']}")
            self._run_cmd([ACE_LM, "--models", ACE_MODELS, "--request", json_path], timeout=1800)
            if llm_json.exists():
                with open(llm_json, 'r', encoding='utf-8') as f:
                    enhanced = json.load(f)
                with open(json_path, 'r', encoding='utf-8') as f:
                    current = json.load(f)
                current.update({k: v for k, v in enhanced.items() if k != "audio_codes" or v})
                current["audio_codes"] = enhanced.get("audio_codes", "")
                current["seed"] = 0
                current["inference_steps"] = INFERENCE_STEPS
                current["synth_model"] = SYNTH_MODEL
                if ref_audio_path and ref_audio_path.exists():
                    current["task_type"] = "text2music"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(current, f, indent=4)
                logger.info(f"Updated track json with generated audio codes: {json_path.name}")
            else:
                logger.warning(f"LLM enhancement output missing ({llm_json.name}); synthesizing without audio codes.")
        except Exception as e:
            logger.warning(f"LLM enhancement failed for '{entry['title']}' ({e}); synthesizing without audio codes.")
        finally:
            llm_json.unlink(missing_ok=True)

        before = {p.name for p in OUTPUT_DIR.glob("*.wav")}
        synth_cmd = [
            ACE_SYNTH, "--models", ACE_MODELS, "--request", json_path,
            "--vae-chunk", "512", "--vae-overlap", "128"
        ]
        if ref_audio_path and ref_audio_path.exists():
            synth_cmd += ["--ref-audio", str(ref_audio_path)]
        self._run_cmd(synth_cmd, timeout=7200)

        candidates = [p for p in OUTPUT_DIR.glob(f"{stem}*.wav") if p.name not in before]
        if not candidates:
            candidates = [p for p in OUTPUT_DIR.glob("*.wav")
                          if p.name not in before and p.is_file()]
        if not candidates:
            raise RuntimeError(f"No WAV file was produced for '{entry['title']}'.")

        wav_out = max(candidates, key=lambda p: p.stat().st_mtime)
        final_wav = OUTPUT_DIR / f"{track_num:02d} - {entry['slug']}.wav"
        if wav_out != final_wav:
            wav_out.replace(final_wav)
        return final_wav

    def _generate_cover(self, prompt):
        png_path = OUTPUT_DIR / "cover.png"
        png_path.unlink(missing_ok=True)
        seed = random.randint(0, 2**32 - 1)
        cmd = [
            SD_CLI,
            "--diffusion-model", SD_MODELS / COVER_MODEL,
            "--llm", SD_MODELS / COVER_LLM,
            "-H", str(COVER_H),
            "-W", str(COVER_W),
            "--vae", SD_MODELS / COVER_VAE,
            "--vae-conv-direct",
            "--sampling-method", "euler",
            "--scheduler", "smoothstep",
            "--steps", str(COVER_STEPS),
            "--cfg-scale", "1",
            "-p", prompt,
            "-s", str(seed),
            "-o", png_path
        ]
        self._run_cmd(cmd, timeout=3600)

        if not png_path.exists():
            raise RuntimeError("sd-cli did not produce an output image.")

        img = Image.open(png_path)
        img = img.convert("RGB")
        jpg_path = OUTPUT_DIR / "cover.jpg"
        img.save(jpg_path, format="JPEG", quality=92)
        png_path.unlink(missing_ok=True)
        logger.info(f"Cover art saved: {jpg_path}")

    def _bundle_zip(self, zip_name, track_entries, wav_paths):
        zip_path = BASE_DIR / zip_name
        zip_path.unlink(missing_ok=True)

        album_name = "Untitled Album"
        album_name_path = OUTPUT_DIR / "album_name.txt"
        if album_name_path.exists():
            album_name = album_name_path.read_text(encoding='utf-8').strip()

        flac_paths = []
        for i, wav_path in enumerate(wav_paths):
            if not wav_path.exists():
                continue
            track_num = i + 1
            entry = track_entries[i] if i < len(track_entries) else None
            slug = entry["slug"] if entry else wav_path.stem
            flac_path = OUTPUT_DIR / f"{track_num:02d} - {slug}.flac"

            try:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(wav_path), "-c:a", "flac", str(flac_path)],
                    capture_output=True, text=True, timeout=600
                )
                if result.returncode != 0:
                    logger.warning(f"FFmpeg conversion failed for {wav_path.name}: {result.stderr[:200]}")
                    continue

                title = entry["title"] if entry else slug
                try:
                    tag = FLAC(str(flac_path))
                    tag["title"] = title
                    tag["album"] = album_name
                    tag["artist"] = "AI Generator"
                    tag["tracknumber"] = str(track_num)
                    tag.save()
                    logger.info(f"Tagged {flac_path.name}")
                except Exception as e:
                    logger.warning(f"Could not tag FLAC {flac_path.name}: {e}")

                cover_jpg = OUTPUT_DIR / "cover.jpg"
                if cover_jpg.exists():
                    try:
                        pic = Picture()
                        pic.type = 3
                        pic.mime = "image/jpeg"
                        pic.data = cover_jpg.read_bytes()
                        pic.width = 768
                        pic.height = 768
                        tag2 = FLAC(str(flac_path))
                        tag2.clear_pictures()
                        tag2.add_picture(pic)
                        tag2.save()
                        logger.info(f"Embedded cover art into {flac_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not embed cover art in {flac_path.name}: {e}")

                flac_paths.append(flac_path)
                logger.info(f"Converted {wav_path.name} -> {flac_path.name}")
            except Exception as e:
                logger.warning(f"Error converting {wav_path.name}: {e}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(OUTPUT_DIR.iterdir()):
                if f.is_file() and f.suffix.lower() in (".json", ".txt", ".flac", ".jpg"):
                    zf.write(f, f.name)
        logger.info(f"Bundled album into {zip_path}")

    def run(self):
        host = _cfg("HOST", "0.0.0.0")
        port = int(_cfg("PORT", "3002"))
        self.app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    app_instance = AlbumGeneratorApp()
    print("Starting Album Generator...")
    print(f"MusicGen backend: {MUSICGEN_DIR} (ace-lm={ACE_LM.exists()}, ace-synth={ACE_SYNTH.exists()}, models={ACE_MODELS.exists()})")
    print(f"SD backend:       {SDGUI_DIR} (sd-cli={SD_CLI.exists()}, models={SD_MODELS.exists()})")
    print(f"Output Directory: {OUTPUT_DIR.resolve()}")
    app_instance.run()
