# Image + Audio Portrait Generator (Retrato Hablado)

A lightweight CLI tool that records voice input, transcribes it, validates/collects facial characteristic data (via a JSON schema), uses text-to-speech to ask follow-up questions in Spanish, and finally generates a portrait image from the collected data using OpenAI models. The project is primarily Spanish-language oriented (prompts, TTS instructions, and expected conversational flow).

> NOTE: This project can be used to create "retrato hablado" style portraits from witness descriptions. This has potentially sensitive and legal implications — read the "Safety & Ethics" section below.

## Features
- Microphone recording and WAV file saving (record until Enter).
- Speech-to-text transcription.
- Moderation check for input.
- Validation/collection of structured characteristics using a Pydantic schema.
- Text-to-speech feedback and question prompts (Spanish).
- Image generation from the final structured description (base64 -> PNG).
- Cost/usage tracking for transcription, text, TTS, image generation, and moderation.

## Quick demo
1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
2. Run the main script:
   ```bash
   python main.py
   ```
3. Speak into your microphone; press Enter to stop recording. The tool transcribes, asks follow-up questions (audio + text), and when all required data is collected it generates `portrait.png`, `data.json`, and expense/usage `tokens.json`. It also writes audio responses to `response.wav` and saves the recorded input as `record.wav`.

## Requirements
- Python 3.10+
- Packages:
  - openai (or the package that provides OpenAI client with `OpenAI` class)
  - pydantic
  - sounddevice
  - scipy
  - numpy
- System audio dependencies (for sounddevice):
  - On Debian/Ubuntu: `sudo apt-get install libportaudio2 portaudio19-dev` (or the appropriate PortAudio package for your OS)
  - On macOS: install portaudio via Homebrew: `brew install portaudio`

## Configuration
- API Key: the script expects `OPENAI_API_KEY` in environment variables (standard OpenAI SDK behavior).
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
- Model names can be changed inside `main.py`:
  - `trans_model` (audio transcription), default: `gpt-4o-transcribe` (from the repo)
  - `text_model` (text processing/validation), default: `gpt-5-nano`
  - `audio_model` (TTS), default: `gpt-4o-mini-tts`
  - `image_model` (image generation), default: `gpt-image-1`
- Image prompt template is in `main.py` (variable `image_instruc`). Edit if you want different styles or instructions.
- Pricing and cost-tracking settings are in `main.py` (`cost_per_1M_tokens` and related logic). Update prices if you want accurate accounting for different model billing.

## Files & Key Components
- `main.py`
  - Primary control loop:
    - Calls `record_audio()` to produce `record.wav`
    - Uses OpenAI client for transcription, moderation, schema-based validation/collection, TTS, and image generation
    - Writes outputs: `data.json` (collected structured data), `portrait.png` (image), `tokens.json` (cost/usage), `response.wav` (TTS playback)
  - `Characteristics` and `ValidateData` Pydantic models define the schema for the portrait data.
  - `track_usage(...)` function centralizes cost tracking and summary.
  - Prompts, instructions, and Spanish-language behavior are defined inline as long strings.
- `audio.py`
  - `record_audio(file_name="temp.wav", sample_rate=44100)`: records from the default microphone and stops when the user presses Enter. Saves WAV file.
  - `play_audio(file_name="temp.wav")`: reads and plays WAV files via sounddevice.
  - Handles basic exceptions (file-not-found, playback errors).
- Outputs created by the program:
  - `record.wav` — captured user audio
  - `response.wav` — TTS responses from the assistant
  - `data.json` — final structured characteristics
  - `portrait.png` — generated image (decoded from base64)
  - `tokens.json` — usage/cost tracking (detailed)

## Usage examples
- Record + run once (interactive):
  ```bash
  python main.py
  ```
  Follow audio prompts. The assistant will ask follow-up questions in Spanish until `datos_completos` is true. When finished it generates an image and saves files to the working directory.

- Tweak TTS voice, speed, or model:
  Edit the TTS call parameters inside `main.py` where `client.audio.speech.create(...)` is used.

- Change output file names:
  Modify the filenames inside `main.py` where `open("portrait.png", "wb")`, `open("data.json", "w")`, etc. are used.

## Models & Costs
- The repo includes a basic cost estimation/tracking mechanism. It uses a `cost_per_1M_tokens` mapping for different models and a per-image cost for image generation. Update those values to match the current pricing of the models you use.
- The code logs token counts and approximate costs in `tokens.json`. This is a simple estimator, not an invoice.

## Acknowledgements
- Built around the OpenAI APIs for transcription, moderation, text parsing/RAG, TTS, and image generation.
- Audio helpers use sounddevice + scipy for recording and playback.
