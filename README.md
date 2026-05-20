# 🎙️ Podcast Autopilot

> **AI-powered podcast summarization pipeline** — transcribe, intelligently trim, and export a condensed highlight reel from any audio file using OpenAI Whisper and GPT-4 Turbo.

---

## 📌 Overview

**Podcast Autopilot** is an end-to-end audio processing pipeline that automatically transforms long-form podcast or lecture recordings into tight, high-signal highlight clips — without touching a single audio editor. You drop in an MP3 or WAV file, set a target length (e.g., 30% of the original), and the system handles everything: transcription, intelligent segment selection, audio cropping, and final export.

The tool is designed for content creators, researchers, and developers who want to extract the most valuable moments from hours of audio — automatically and at scale.

---

## ✨ Features

- **Automatic Speech-to-Text** — Transcribes audio using OpenAI Whisper with word-level timestamps
- **Timestamped Transcript Formatting** — Groups words into readable lines, split on natural speech pauses (>1 second)
- **AI-Driven Segment Selection** — Uses GPT-4 Turbo to identify and select only the highest-value segments from the transcript
- **Target Duration Control** — Specify an exact percentage of the original runtime to keep (default: 30%)
- **Retry & Validation Logic** — Automatically re-queries the model if the selected duration is off-target, ensuring accuracy within ±10%
- **Audio Cropping & Stitching** — Extracts the selected segments and joins them with smooth 800ms silence padding
- **Dual Export** — Outputs both a high-quality WAV and a 192kbps MP3 of the final edited audio
- **Django-Ready Utility Module** — `utils.py` is structured for integration into a Django web application with settings-based API key management
- **WAV → MP3 Auto-Conversion** — Accepts both WAV and MP3 inputs; WAV files are automatically converted before processing

---

## 🗂️ Project Structure

```
podcast-autopilot/
│
├── main.py                  # Standalone pipeline script with retry/validation logic
├── utils.py                 # Django-integrated version of the pipeline
├── test_final_1.py          # Test script variant 1
├── test_final_2.py          # Test script variant 2
├── requirements.txt         # All Python dependencies
├── .gitignore
│
├── Bishop Varden Lecture.mp3  # Sample audio file
└── Tucker.mp3                 # Sample audio file
```

---

## 🔄 How It Works

```
Audio File (MP3/WAV)
        │
        ▼
  [1] Whisper Transcription
      (word-level timestamps)
        │
        ▼
  [2] Timestamp Formatting
      (pause-aware line grouping)
        │
        ▼
  [3] GPT-4 Turbo Summarization
      (select key segments → JSON)
        │
        ▼
  [4] Duration Validation & Retry
      (re-query if off target by >10%)
        │
        ▼
  [5] Audio Cropping & Stitching
      (pydub segment extraction)
        │
        ▼
  [6] Export: WAV + MP3
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.x** | Core language |
| **OpenAI Whisper** (`whisper-1`) | Speech-to-text transcription with word timestamps |
| **GPT-4 Turbo** | Intelligent segment selection and summarization |
| **pydub** | Audio cropping, stitching, and format conversion |
| **ffmpeg** | Audio backend for pydub |
| **python-dotenv** | Environment variable management |
| **Django** | Web framework integration (via `utils.py`) |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/download.html) installed and available in your system PATH
- An OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/BusraRafa/podcast-autopilot.git
cd podcast-autopilot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🚀 Usage

### Standalone Script (`main.py`)

Edit the bottom of `main.py` to point to your audio file:

```python
if __name__ == "__main__":
    audio_file = "./your_podcast.mp3"
    output_folder = "./output"

    result = process_audio_pipeline(audio_file, output_folder, target_percentage=30)
```

Then run:

```bash
python main.py
```

### Function Signature

```python
process_audio_pipeline(
    audio_file_path: str,   # Path to your MP3 or WAV file
    output_folder: str,     # Directory for all output files
    target_percentage: int  # % of original duration to keep (default: 30)
)
```

### Output Files

After running, the `output_folder` will contain:

| File | Description |
|---|---|
| `demo_transcription_formatted_output.txt` | Full timestamped transcript |
| `output.json` | JSON array of selected segments with timestamps |
| `<name>_FINAL_EDITED.wav` | Final highlight reel (WAV) |
| `<name>_FINAL_EDITED.mp3` | Final highlight reel (MP3, 192kbps) |

---

## 📦 Key Dependencies

```
openai==2.9.0
openai-whisper==20250625
pydub==0.25.1
ffmpeg-python==0.2.0
python-dotenv==1.2.1
torch==2.9.1
```

See `requirements.txt` for the full list.

---

## 🧠 Design Decisions

**Why keep exact wording from the transcript?**
The pipeline instructs GPT to never paraphrase or modify source text — all selected segments are verbatim excerpts. This ensures the cropped audio matches the selected text exactly, making the JSON-to-audio alignment reliable.

**Why retry logic?**
LLMs don't always produce outputs of a precise length on the first attempt. The pipeline calculates the total duration of selected segments after each response and retries with a stricter prompt if the result deviates more than 10% from the target.

**Why 800ms silence padding?**
Short silence gaps between stitched segments make the final audio sound natural rather than abruptly cut. This value is configurable in the code.

---

## 🔮 Potential Extensions

- Web UI via Django for drag-and-drop audio upload
- Support for YouTube URL input (via `yt-dlp`)
- Chapter-aware summarization for structured podcasts
- Speaker diarization to preserve only a specific speaker
- Batch processing for entire podcast RSS feeds

---

## 📄 License

This project is open source. Feel free to use, modify, and build upon it.

---

*Built with OpenAI Whisper + GPT-4 Turbo + pydub*
