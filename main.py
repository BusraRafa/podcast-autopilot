from pydub import AudioSegment
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def process_audio_pipeline(audio_file_path: str, output_folder: str, target_percentage: int = 30):
    """
    Complete audio processing pipeline with strict duration targeting:
    1. Transcribe audio using Whisper
    2. Format transcription with timestamps
    3. Summarize transcript using GPT-4 Turbo (targeting specific percentage)
    4. Validate and retry if duration target not met
    5. Crop and join audio segments
    6. Export final WAV and MP3
    
    Args:
        audio_file_path: Path to input audio file
        output_folder: Directory where all outputs will be saved
        target_percentage: Target percentage of original duration (default: 30)
    
    Returns:
        dict: Contains paths to final WAV and MP3 files
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    
    audio_file = open(audio_file_path, "rb")
    
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    
    audio_file.close()
    
    
    original_audio = AudioSegment.from_file(audio_file_path)
    original_duration_seconds = len(original_audio) / 1000
    target_duration_seconds = original_duration_seconds * (target_percentage / 100)
    
    print(f" Original Duration: {original_duration_seconds/60:.2f} minutes")
    print(f" Target Duration: {target_duration_seconds/60:.2f} minutes ({target_percentage}%)")
    
    
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
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    
    original_text = read_text_file(transcript_file)
    
    
    system_prompt = "You are an expert transcript editor specializing in extracting ONLY the absolute most critical content from conversations. You always return valid JSON arrays with no additional text."
    
    user_prompt = f"""
INPUT TRANSCRIPT DETAILS:
- Original Duration: {original_duration_seconds/60:.2f} minutes
- Target Duration: {target_duration_seconds/60:.2f} minutes ({target_percentage}% of original)

YOUR MISSION:
✔ Select ONLY the **most impactful, unique, and essential moments** - the absolute CORE of the conversation
✔ Be EXTREMELY SELECTIVE - remove ALL:
  • Repetitive points (even if worded differently)
  • Filler words, pleasantries, small talk
  • Introductions, outros, and transitional phrases
  • Rambling or tangential discussions
  • Background music segments or non-speech audio
  • ANY content that doesn't add unique value
  
✔ Keep ONLY segments with:
  • Key insights or "aha" moments
  • Critical information or main arguments
  • Powerful quotes or memorable statements
  • Essential context needed to understand the core message

✔ **STRICT REQUIREMENT**: Your selected segments MUST total approximately {target_duration_seconds/60:.2f} minutes ({target_percentage}% of {original_duration_seconds/60:.2f} minutes)

✔ **DO NOT modify ANY original words** - keep exact wording from transcript
✔ Each segment must have timestamp format: [MM:SS-MM:SS]

OUTPUT FORMAT (JSON only):
[
  {{
    "timestamp": "[00:00-00:05]",
    "text": "Exact original text from transcript"
  }},
  {{
    "timestamp": "[00:10-00:15]",
    "text": "Another critical excerpt exactly as written"
  }}
]

 CRITICAL: Calculate the total duration of your selected segments. They should add up to approximately {target_duration_seconds:.0f} seconds ({target_duration_seconds/60:.2f} minutes).

--- TRANSCRIPT START ---
{original_text}
--- TRANSCRIPT END ---

Return ONLY the JSON array. No explanations, no extra text.
"""
    
    
    max_retries = 2
    best_result = None
    best_duration_diff = float('inf')
    
    for attempt in range(max_retries + 1):
        
        
        
        current_user_prompt = user_prompt
        if attempt > 0:
            current_user_prompt += f"\n\n RETRY: Previous attempt was too long. Be MORE selective. Target EXACTLY {target_duration_seconds/60:.2f} minutes."
        
        try:
            
            response = client.chat.completions.create(
                model="gpt-4-turbo",  # or "gpt-4-turbo-2024-04-09" or "gpt-4o"
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_user_prompt}
                ],
                temperature=0.7,
                #max_tokens=4000
            )
            
            summary_result = response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"    API Error: {e}")
            continue
        
        
        if summary_result.startswith("```"):
            summary_result = re.sub(r'```json\s*|\s*```', '', summary_result).strip()
        
        
        try:
            json_data = json.loads(summary_result)
            
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
            
            total_duration_ms = 0
            for entry in json_data:
                start_ms, end_ms = parse_timestamp(entry.get("timestamp", ""))
                total_duration_ms += (end_ms - start_ms)
            
            
            total_duration_ms += (len(json_data) - 1) * 800  
            
            actual_duration_seconds = total_duration_ms / 1000
            duration_diff = abs(actual_duration_seconds - target_duration_seconds)
            
            
            
            if duration_diff < best_duration_diff:
                best_duration_diff = duration_diff
                best_result = summary_result
            
            
            percentage_achieved = (actual_duration_seconds / original_duration_seconds) * 100
            if abs(percentage_achieved - target_percentage) <= 10:
                
                break
                
        except Exception as e:
            print(f" Error parsing result: {e}")
            continue
    
    if not best_result:
        print("Failed to generate valid summary")
        return None
    
    
    json_output_path = os.path.join(output_folder, "output.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        f.write(best_result)
    
    
    
    
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    wav_path = os.path.join(output_folder, f"{base_name}_temp_converted.wav")
    
    audio = AudioSegment.from_file(audio_file_path)
    audio.export(wav_path, format="wav")
    audio = AudioSegment.from_wav(wav_path)
    
    with open(json_output_path, "r", encoding="utf-8") as f:
        json_data = json.loads(best_result)
    
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
            
            if i < len(json_data) and add_silence_ms > 0:
                combined += AudioSegment.silent(duration=add_silence_ms)
        
        except Exception as e:
            print(f"Skipping segment {i}: {e}")
            pass
    
    
    final_wav = os.path.join(output_folder, f"{base_name}_FINAL_EDITED.wav")
    final_mp3 = os.path.join(output_folder, f"{base_name}_FINAL_EDITED.mp3")
    
    combined.export(final_wav, format="wav")
    combined.export(final_mp3, format="mp3", bitrate="192k")
    
    final_duration_minutes = len(combined) / 1000 / 60
    final_percentage = (len(combined) / len(original_audio)) * 100
    

    
    return {
        "transcript": transcript_file,
        "summary_json": json_output_path,
        "final_wav": final_wav,
        "final_mp3": final_mp3,
        "original_duration_minutes": original_duration_seconds / 60,
        "final_duration_minutes": final_duration_minutes,
        "percentage_achieved": final_percentage
    }


if __name__ == "__main__":
    audio_file = "./The_20Meaning_20Of__20There_27s_20No_20Such_20Thing_20as_20a_20Dragon_20__20EP_20566_20_T2fz9ZhmaQA_ (1).mp3"
    output_folder = "C:/Users/Busra/podcast/test_output"
    
 
    result = process_audio_pipeline(audio_file, output_folder, target_percentage=30)
    