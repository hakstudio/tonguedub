# Tonguedub

Local **English → Turkish video dubbing** that keeps each speaker's own voice.
Tonguedub separates the speech from the background, transcribes it, translates it with an LLM,
re-speaks every line in Turkish with a **cloned voice** of the original speaker, and mixes it back
into the video. Everything runs on your Mac except the translation step (text only → DeepSeek API).
Audio and video never leave your machine.

> Tested on Apple Silicon (M2, 24 GB). Other platforms are not supported.
> The command-line messages are currently in Turkish.

## Pipeline

1. **ffmpeg + demucs**: extract audio, split vocals / background.
2. **WhisperKit** (CoreML, large-v3-turbo): word-timed transcript, split into sentences.
3. **DeepSeek API**: context-aware, duration-aware translation; lines that would not fit are shortened.
4. **Chatterbox Multilingual TTS** (MPS): speakers are grouped by pitch and voice embedding, each is cloned
   from a clean reference clip; up to 3 parallel workers share one job pool.
5. **Mix**: fit clips so they do not overlap (speed-up / push / fade), restore pitch and level per speaker,
   duck the background.

## Install

Requirements: macOS on Apple Silicon, [Homebrew](https://brew.sh), ~10 GB free disk.

```sh
git clone <repo-url> tonguedub && cd tonguedub
./scripts/install.sh      # brew deps, Python 3.11 venv, packages, ~/.local/bin/tonguedub shortcut
# edit .env and put your DeepSeek API key into it (install.sh creates it from .env.example)
tonguedub doctor             # checks tools, models, API key
```

Settings live in `config.toml` (created from `config.example.toml` by `install.sh`, as is `.env` from `.env.example`); **secrets live in `.env`**
(`DEEPSEEK_API_KEY=...`). The Whisper model is downloaded on first use unless you set `[transcribe] model_path`.

## Usage

```sh
tonguedub video.mp4                       # → video_TR.mp4 next to the original
tonguedub video.mp4 --start 0 --end 120   # quick 2-minute trial
tonguedub video.mp4 --merge 20            # batch short lines (faster, slightly riskier)
tonguedub video.mp4 --test                # keep intermediate files for inspection

tonguedub add a.mp4 b.mp4 --merge 20      # queue: preparation of the next video overlaps TTS of the current one
tonguedub status | clear | clean
tonguedub run --help                      # all options
```

Options override `config.toml`; every option has a matching key in `config.example.toml`.

## Configuration

| File | Holds | In git? | Template |
|---|---|---|---|
| `config.toml` | paths, models, pipeline options | no | `config.example.toml` |
| `.env` | secrets (`DEEPSEEK_API_KEY`) | no | `.env.example` |

`config.toml` is looked up in this order: `--config` → `$TONGUEDUB_CONFIG` → `<repo>/config.toml` → `~/.config/tonguedub/config.toml`.
`.env` is looked up in `<repo>/.env`, then `~/.config/tonguedub/.env`; a real environment variable always wins.
Unknown keys or wrong types are rejected with a clear error, and secrets are deliberately not accepted in `config.toml`.
`tonguedub doctor` warns if `.env` is readable by other users.

## Privacy

- Intermediate files (transcript, translation, audio) live in `[paths] work_dir` (mode 700, excluded from Spotlight)
  and are **deleted when the run ends, fails, or is interrupted**. `--test` keeps them.
- Logs (`[paths] log_dir`) contain timings and counters only, never transcript text.
- Only the English text is sent to the translation API.
- Leftovers from a hard kill (SIGKILL / power loss) can be removed with `tonguedub clean`.

## Responsible use

Tonguedub clones real people's voices. **You are solely responsible for how you use it.** Only dub material you have the
right to process, only clone voices you have permission to clone, and follow the laws and platform rules that apply to you
(copyright, personality/voice rights, consent, disclosure of synthetic media). Do not use it to deceive, impersonate or harm anyone.
The authors provide the software as is, without warranty, and accept no liability for how it is used (see `LICENSE`).

## License

[MIT](LICENSE). You may use, modify and redistribute Tonguedub, including commercially, provided you keep the copyright
notice and license text in copies or substantial portions of the software, which means credit must be retained
wherever the code is reused. The software comes with no warranty and the authors are not liable for any damages or claims
arising from its use.

Third-party components keep their own licenses and are not bundled: Chatterbox TTS, Demucs, WhisperKit and the Whisper
models, PyTorch, librosa and others are installed from their upstream sources; ffmpeg and Rubber Band are separate
system tools installed through Homebrew. Check their terms (and the DeepSeek API terms) for your use case.

## Development

```sh
.venv/bin/pytest          # unit tests (no models, audio or network needed)
.venv/bin/ruff check .
```

Layout: `src/tonguedub/` (one module per pipeline stage: `transcribe`, `translate`, `speakers`, `tts`, `mix`,
orchestration in `pipeline`, queue in `scheduler`, settings in `config`), `tests/`, `scripts/install.sh`.
Measurements and lessons learned are in `CLAUDE.md`.
