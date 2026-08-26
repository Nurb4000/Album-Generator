# Album Generator

One-stop web UI that generates a **complete album**: song planning, music synthesis,
and cover art — bundled into a single zip you download.

Describe a style, pick a track count and target length, hit **Generate Album**, and
the app runs the whole pipeline for you:

```
1. Plan      -> LLM writes unique song titles, lyrics/captions per track,
                 an album name (album_name.txt), and a cover-art prompt
                 (album_cover.txt)
2. Music     -> for each track: ace-lm generates audio codes (merged back into
                 the track's local json), then ace-synth renders the audio
3. Cover     -> sd-cli renders the album cover at 768x768 -> cover.jpg
4. Bundle    -> WAV files are converted to FLAC (with metadata tags), then
                 everything zipped as <Album Name>.zip; download manually via
                 the Download ZIP button (re-downloadable). Files stay on the
                 server until the next run starts, which wipes them.
```

Live progress is shown in the UI (current stage, percent, and which track is
rendering), so you always know it is working. Once a track finishes generating,
a native HTML5 audio player appears inline so you can preview it directly in
the browser before downloading the full album.

## Changelog

### File naming format (track files)
- Track JSON and audio files now use the format: `NN - Song Title.json` / `NN - Song Title.flac`
  where `NN` is the zero-padded track number.
- Words in titles are separated by spaces (not underscores).
- Both JSON and WAV/FLAC files share the same naming convention, including the track number prefix.

### Audio preview in the UI
- Each track now has a native HTML5 `<audio>` player that appears once generation completes.
- Players use the browser's built-in controls (play/pause, seek, volume).
- Served via `/preview/audio/<filename>` — no server-side streaming libraries required.

### FLAC conversion & metadata tagging
- WAV renderings are converted to FLAC using `ffmpeg` before bundling.
- FLAC files are tagged with:
  - `title`   = track name
  - `album`   = album name
  - `tracknumber` = track position (e.g. "1", "2", ...)
  - Coverart added to FLAC metadata
- WAV files are NOT included in the final zip — only FLAC, JSON, TXT, and JPG.

### Negative prompt
- A default negative prompt is applied during synthesis to discourage:
  bad audio quality, robotic vocals, autotune, distortion, spoken word,
  overly loud backing vocals, MIDI artifacts, mechanical piano, glitchy drums,
  overcompression, muddy mix/bass, heavy reverb, crowd noise, background noise,
  unwanted silence, chaotic arrangement, predictable loops, and repetitiveness.

## Dependencies

This app drives two external backends by shelling out to their binaries:

| Backend | Project | Binaries used | Purpose |
|---|---|---|---|
| Music | [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) | `bin/ace-lm`, `bin/ace-synth` + GGUF models | LLM enhancement (audio codes) and music synthesis |
| Images | [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | `bin/sd-cli` + models | Album cover generation |

Both backend folders are expected to have this layout (the same convention used
by the individual GUIs):

```
<backend dir>/
    bin/        <- compiled binaries
    models/     <- GGUF / safetensors model files
```

Defaults point at `/opt/musicgen` and `/opt/sd-gui` (see `.env` below).

**System dependency:** `ffmpeg` must be installed and available on `$PATH` for
WAV-to-FLAC conversion during the bundling step.

## Install & Run

```bash
pip install -r requirements.txt   # flask, requests, pillow, python-dotenv, mutagen
python3 app.py                    # serves on 0.0.0.0:3002
```

## Configuration (.env)

All settings live in the `.env` file next to `app.py`. Values shown are the
defaults. Variables exported in your shell override the file.

```ini
LLM_URL=http://10.0.1.27:8080/       # OpenAI-compatible LLM for planning songs/names/prompts
MUSICGEN_DIR=/opt/musicgen           # acestep.cpp deployment (bin/ + models/)
SDGUI_DIR=/opt/sd-gui                # stable-diffusion.cpp deployment (bin/ + models/)
HOST=0.0.0.0
PORT=3002

TARGET_DURATION_DEFAULT=180          # default "target track length" in the UI
DURATION_JITTER=0.08                 # +/- variation around the target (per track)

SYNTH_MODEL=acestep-v15-xl-turbo-Q8_0.gguf
LM_MODEL=acestep-5Hz-lm-4B-Q8_0.gguf
INFERENCE_STEPS=10

COVER_WIDTH=768
COVER_HEIGHT=768
COVER_STEPS=10
COVER_MODEL=z_image_turbo-Q8_0.gguf
COVER_LLM=qwen_3_4b-Q8_0.gguf
COVER_VAE=ae.safetensors
```

## Generation settings

Track JSONs are created with fixed production-friendly defaults:

- `seed: 0` (random each run)
- `inference_steps: 10`
- synth model: `acestep-v15-xl-turbo-Q8_0.gguf` (turbo = fast)
- durations land within ~+/-10% of your target length so the album has natural
  variety without drifting far from what you asked for

Cover art uses z-image-turbo at 768x768, no lora.

## Output

A single `<Album Name>.zip` containing:

```
album_name.txt            the album name
album_cover.txt           the Stable Diffusion prompt used for the cover
cover.jpg                 768x768 album art
NN - Song Title.json      final ACE-Step request json per track
                          (includes generated audio_codes)
NN - Song Title.flac      converted audio with metadata tags
                          (title, album, track number)
```

The zip downloads automatically when the album finishes; temp files on the
server are deleted right after. If a download fails, nothing is deleted — just
click **Download ZIP** again.

## Related projects

- [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) — music backend
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — image backend
- [acestep.cpp-simple-GUI](https://github.com/Nurb4000/acestep.cpp-simple-GUI) — manual single-song GUI (the music side of this pipeline, interactive)
- [StableDiffusion.CPP-GUI](https://github.com/Nurb4000/StableDiffusion.CPP-GUI) — manual image-generation GUI

Screenshot:

<img width="553" height="798" alt="image" src="https://github.com/user-attachments/assets/6f6f4bfb-3baf-4e1d-b8d6-35fb86408488" />


