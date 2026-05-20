from pydub import AudioSegment
import os
import json
import re

from openai import OpenAI

from django.conf import settings


OPENAI_API_KEY = settings.OPENAI_API_KEY

def process_audio_pipeline(audio_file_path: str, output_folder: str):
    """
    Complete audio processing pipeline:
    1. Transcribe audio using Whisper
    2. Format transcription with timestamps
    3. Summarize transcript using GPT-5
    4. Crop and join audio segments
    5. Export final WAV and MP3
    
    Args:
        audio_file_path: Path to input audio file (e.g., "./Tucker.mp3" or "./Tucker.wav")
        output_folder: Directory where all outputs will be saved
    
    Returns:
        dict: Contains paths to final WAV and MP3 files
    """
    
    os.makedirs(output_folder, exist_ok=True)

    # NEW: if input is WAV, convert to MP3 first and use that MP3 for the rest
    original_input_path = audio_file_path  # keep for naming if you want
    base_name_no_ext = os.path.splitext(os.path.basename(original_input_path))[0]

    ext = os.path.splitext(audio_file_path)[1].lower()
    if ext == ".wav":
        # Convert WAV -> MP3 in output_folder
        temp_mp3_path = os.path.join(output_folder, f"{base_name_no_ext}_converted.mp3")
        audio_wav = AudioSegment.from_wav(audio_file_path)
        audio_wav.export(temp_mp3_path, format="mp3")
        audio_file_path = temp_mp3_path  # from now on, pipeline works on MP3

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Use MP3 (or original non-WAV) for transcription
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )

    def format_time(seconds):
        """Convert seconds to (m:ss) or (h:mm:ss) format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = int(seconds % 60)
        
        if hours > 0:
            return f"({hours}:{minutes:02}:{sec:02})"
        else:
            return f"({minutes}:{sec:02})"
    
    def format_transcription_lines(transcription):
        """Format transcription with timestamps, breaking on pauses > 1 sec"""
        lines = []
        previous_end = 0
        buffer_words = []
        buffer_start = None
        
        for word in transcription.words:
            start = word.start
            end = word.end
            
            # If there's a pause > 1s, start a new line
            if buffer_start is not None and start - previous_end > 1.0:
                timestamp = format_time(buffer_start)
                text = " ".join(buffer_words)
                lines.append(f"{timestamp} {text}")
                
                buffer_words = []
                buffer_start = start
            
            if buffer_start is None:
                buffer_start = start
            
            buffer_words.append(word.word)
            previous_end = end
        
        if buffer_words:
            timestamp = format_time(buffer_start)
            text = " ".join(buffer_words)
            lines.append(f"{timestamp} {text}")
        
        return lines
    
    formatted_lines = format_transcription_lines(transcription)
    
    transcript_file = os.path.join(output_folder, "demo_transcription_formatted_output.txt")
    with open(transcript_file, "w", encoding="utf-8") as f:
        for line in formatted_lines:
            f.write(line + "\n")
    
    def read_text_file(file_path: str) -> str:
        """Read and return the content of a text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    
    original_text = read_text_file(transcript_file)
    
    prompt = f"""
You are a transcript summarizer.

You will be given a transcript that contains timestamps and text.

Your job:

✔ Read the entire transcript carefully.
✔ Select **ONLY the most important and meaningful parts**.
✔ Remove any repetitive, filler, or non-important sections.
✔ **DO NOT modify or rewrite ANY of the original words.**
✔ Keep the original wording exactly as it appears.
✔ Each selected segment must include a timestamp in the format [MM:SS-MM:SS].
✔ Output must be a clean JSON list, like:

[
  {{
    "timestamp": "[00:00-00:05]",
    "text": "Original text from transcript without change."
  }},
  {{
    "timestamp": "[00:10-00:20]",
    "text": "Another important excerpt exactly as in transcript."
  }}
]

THE MOST IMPORTANT REQUIREMENT:
→ Perform stronger summarization.
→ Keep only the top important segments.
→ All text in the output must appear EXACTLY in the original transcript.
→ Do not add, modify, or paraphrase any text.

Now process the transcript below:

--- TRANSCRIPT START ---
{original_text}
--- TRANSCRIPT END ---

Return ONLY the final JSON list. No explanations.
"""
    
    response = client.responses.create(
        model="gpt-5",  # gpt-5.1-2025-11-13
        input=prompt
    )
    
    summary_result = response.output_text.strip()
    
    if summary_result:
        json_output_path = os.path.join(os.path.dirname(transcript_file), "output.json")
        
        with open(json_output_path, "w", encoding="utf-8") as f:
            f.write(summary_result)
    else:
        return None
    
    def parse_timestamp(timestamp: str):
        """Convert [MM:SS-MM:SS] → (start_ms, end_ms)"""
        pattern = r"\[(\d{2}):(\d{2})-(\d{2}):(\d{2})\]"
        match = re.match(pattern, timestamp.strip())
        if not match:
            raise ValueError(f"Invalid timestamp format: {timestamp}")
        start_min, start_sec, end_min, end_sec = map(int, match.groups())
        start_ms = (start_min * 60 + start_sec) * 1000
        end_ms = (end_min * 60 + end_sec) * 1000
        return start_ms, end_ms
    
    # Use original base name for outputs, not the _converted one
    base_name = base_name_no_ext
    wav_path = os.path.join(output_folder, f"{base_name}_temp_converted.wav")
    
    # Load the (possibly converted) MP3 and convert to WAV for segment editing
    audio = AudioSegment.from_file(audio_file_path)
    audio.export(wav_path, format="wav")
    
    audio = AudioSegment.from_wav(wav_path)

    with open(json_output_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    combined = AudioSegment.empty()
    add_silence_ms = 800  
    
    for i, entry in enumerate(json_data, start=1):
        try:
            timestamp = entry.get("timestamp")
            if not timestamp:
                continue
            
            start_ms, end_ms = parse_timestamp(timestamp)
            end_ms = min(end_ms, len(audio))  
            
            segment = audio[start_ms:end_ms]
            
            combined += segment
            if add_silence_ms > 0:
                combined += AudioSegment.silent(duration=add_silence_ms)
        
        except Exception:
            # You might want to log the error here
            pass
    
    final_wav = os.path.join(output_folder, f"{base_name}_FINAL_EDITED.wav")
    final_mp3 = os.path.join(output_folder, f"{base_name}_FINAL_EDITED.mp3")
    
    combined.export(final_wav, format="wav")
    combined.export(final_mp3, format="mp3", bitrate="192k")
    
    return {
        "transcript": transcript_file,
        "summary_json": json_output_path,
        "final_wav": final_wav,
        "final_mp3": final_mp3,
        "duration_minutes": len(combined) / 1000 / 60
    }


if __name__ == "__main__":
    audio_file = "./Tucker.mp3"  # or "./Tucker.wav"
    output_folder = "C:/Users/Busra/podcast/test_output"
    
    result = process_audio_pipeline(audio_file, output_folder)
    
    if result:
        print(f"Duration: {result['duration_minutes']:.2f} minutes")