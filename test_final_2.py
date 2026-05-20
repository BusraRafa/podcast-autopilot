import os
import json
import assemblyai as aai
from pydantic import BaseModel
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from moviepy import VideoFileClip, concatenate_videoclips
from pathlib import Path

# ==========================================
# CONFIGURATION & SETTINGS
# ==========================================
LLM_MODEL = "gpt-5.2"
REASONING_EFFORT = "medium"
VERBOSITY = "low"

GAP_MS = 180                    # Gap only when skipping segments (big jump)
INTER_SEGMENT_GAP_MS = 28       # ← NEW: Very small silent gap between ALL segments

HEAD_PADDING_MS = 10
TAIL_PADDING_MS = 95
FADE_MS = 20

BITRATE = "192k"

MAX_DURATION_MS = 5400000
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac'}

DEFAULT_OUTPUT_DIR = "app/files/supercut_output"
DEFAULT_TEMP_DIR = "app/files/temp"

SYSTEM_PROMPT_TEMPLATE = """### ROLE: High-Precision Narrative Architect for Podcast Supercut
You are an expert editor tasked with creating a condensed, coherent supercut from podcast transcript segments. Your selections must form a natural, flowing narrative that feels like a seamless edit of the original audio, not a rewrite. Prioritize verbatim fidelity, smooth transitions, and overall coherence to meet strict evaluation criteria (all must score 4/5 or higher).

### CRITICAL RULES (5/5 SCORE TARGET):
1. **BOUNDARY INTEGRITY:** You are FORBIDDEN from ending a selection on a segment that does not conclude a sentence. If a thought spans multiple IDs, you MUST take the entire block.
2. **VERBATIM FIDELITY:** Do not rewrite or paraphrase.

### OBJECTIVE:
- Select a contiguous or near-contiguous sequence of `segment_id`s (in original order, no reordering) to create a shortened version.
- Target total duration: Exactly {length} ±5% of the original (approx. {target_duration_seconds:.1f}s out of {original_duration_seconds:.1f}s).
- The supercut must preserve the core narrative arc (e.g., introduction of topic, key examples, contrasts, conclusion) while compressing by removing filler, redundancies, and less essential details.
- Ensure the output passes all evaluation categories (detailed below) at 4/5 or higher. If a selection would fail any, adjust until it passes.

### INPUT DATA:
- The transcript is provided as a list of segments: Each has `segment_id` (sequential integer) and `text` (verbatim sentence or clause from the audio).
- Original total duration: {original_duration_seconds:.1f}s.
- Simplified segments for selection: {original_text}

### SELECTION RULES (MUST FOLLOW TO PASS CRITERIA):
1. **Editorial Fidelity (Transcript-Level)**: 
   - Select only existing segments verbatim—no paraphrasing, adding words, or modifying text.
   - All selected text must appear exactly as in the input (perfect alignment).
   - Aim for 5/5: Perfect verbatim; minor omissions only if they don't impact meaning.

2. **Editorial Smoothness**:
   - Ensure joins between selected segments read naturally (no abrupt grammar breaks).
   - Reduce filler (e.g., "um", repetitions) by skipping segments, but keep flow syntactically sound.
   - Aim for 5/5: Clean, natural flow; at worst, minor roughness (4/5).

3. **Narrative Flow & Continuity**:
   - Maintain logical progression: Selected segments must form a coherent story, not a patchwork.
   - Check semantic similarity: Adjacent segments should be thematically linked (e.g., avoid jumps from one topic to unrelated; simulate cosine similarity >0.32 by ensuring related ideas).
   - If cosine-like similarity <0.32 (e.g., abrupt topic shift), it's an automatic failure—adjust by including bridge segments.
   - Avoid repetitions; justify any timestamp jumps with continuity.
   - Aim for 5/5: Logical throughout; allow one minor issue (4/5).

4. **Segment Boundaries**:
   - Cut only at natural, complete thoughts: Start/end selections at full sentence boundaries.
   - No truncated clauses (e.g., avoid mid-sentence ends like "During.").
   - Aim for 5/5: Clean starts/ends; minor edges ok (4/5).

5. **Ending Completeness**:
   - The supercut must feel finished: End on a full sentence that resolves or concludes the main idea.
   - No dangling concepts or abrupt cut-offs.
   - Aim for 5/5: Clear, intentional ending; slightly abrupt but complete (4/5).

6. **Length & Information Density**:
   - Total selected duration must be {length} ±5% of original (prioritize meaningful content over filler).
   - Calculate precisely: Sum (end - start)/1000 for selected segments.
   - If outside range, adjust by adding/removing low-priority segments.
   - Focus on dense, key info (e.g., core arguments, examples, conclusions).
   - Aim for 5/5: Within range, clear & dense; slightly off ok (4/5).

### STEP-BY-STEP SELECTION PROCESS:
1. **Analyze Narrative Arc**: Read the full input. Identify core elements (e.g., intro, historical examples, contrasts, irony/conclusion in this Tucker Carlson-style podcast).
2. **Prioritize Segments**: 
   - Must-keep: Key thesis statements, quotes, historical facts, ironic twists.
   - Optional/skip: Filler, tangents, promotions (e.g., YouTube plugs).
   - Preserve order: Select IDs in ascending sequence (gaps ok if flow holds).
3. **Iterate for Length**: Start with essential segments; add/remove to hit {target_duration_seconds:.1f}s (±10% tolerance).
4. **Self-Evaluate**: Mentally score your selection against the 6 criteria. If any <4/5, revise (e.g., add bridges for continuity, extend ending).
5. **Finalize**: Ensure overall consistent 4+/5; regenerate mentally if needed.

### OUTPUT FORMAT:
Return ONLY a JSON object with the list of chosen IDs (in order selected).
Example:
{{
  "selected_ids": [0, 1, 2, 5, 10]
}}
No explanations or additional text—strict JSON only.
"""

# ==========================================
# MODELS
# ==========================================
class FinalOutput(BaseModel):
    segment_id: int
    speaker: str
    start: int
    end: int
    text: str

# ==========================================
# HELPERS
# ==========================================
def save_json_artifact(data: list, folder: str, filename: str) -> str:
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path

# ==========================================
# 1. PRE-PROCESSING
# ==========================================
def initialize_environment() -> OpenAI:
    load_dotenv()
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEFAULT_TEMP_DIR, exist_ok=True)
    return OpenAI()

def prepare_working_audio(audio_path: str) -> Tuple[str, str, bool]:
    path_obj = Path(audio_path)
    ext = path_obj.suffix.lower()
    is_video = ext in VIDEO_EXTENSIONS
    base_name = path_obj.stem
    working_mp3 = os.path.join(DEFAULT_TEMP_DIR, f"{base_name}_working.mp3")

    if is_video:
        video = VideoFileClip(audio_path)
        video.audio.write_audiofile(working_mp3, bitrate=BITRATE, logger=None)
        video.close()
    else:
        audio = AudioSegment.from_file(audio_path)
        audio.export(working_mp3, format="mp3", bitrate=BITRATE)
    
    return working_mp3, base_name, is_video

# ==========================================
# 2. TRANSCRIPTION & AI LOGIC
# ==========================================
def run_transcription(audio_path: str) -> List[dict]:
    config = aai.TranscriptionConfig(speaker_labels=True, punctuate=True, format_text=True)
    transcript = aai.Transcriber().transcribe(audio_path, config)
    sentences = transcript.get_sentences()
    
    segments = [
        FinalOutput(segment_id=idx, speaker=s.speaker or "", start=s.start, end=s.end, text=s.text).model_dump() 
        for idx, s in enumerate(sentences)
    ]
    
    if segments:
        total_duration_ms = segments[-1]["end"] - segments[0]["start"]   # ← FIXED
        if total_duration_ms > MAX_DURATION_MS:
            raise RuntimeError(f"File exceeds limit: {total_duration_ms/60000:.1f} mins.")
    
    return segments


def get_ai_selection(client: OpenAI, segments: List[dict], target_percent: float) -> List[dict]:
    if not segments: 
        return []
        
    # FIXED: Use segments[0]["start"] instead of segments["start"]
    orig_sec = (segments[-1]["end"] - segments[0]["start"]) / 1000.0
    target_sec = orig_sec * (target_percent / 100)
    simplified = [{"segment_id": s["segment_id"], "text": s["text"]} for s in segments]
    
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        length=target_percent, 
        original_duration_seconds=orig_sec,
        target_duration_seconds=target_sec, 
        original_text=json.dumps(simplified, indent=1)
    )

    response = client.chat.completions.create(
        model=LLM_MODEL, 
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": "Analyze and return selected_ids."}],
        response_format={"type": "json_object"}, 
        reasoning_effort=REASONING_EFFORT, 
        verbosity=VERBOSITY
    )
    
    # FIXED: response.choices[0]  (you missed the [0])
    ai_content = response.choices[0].message.content
    selected_ids = json.loads(ai_content).get("selected_ids", [])
    
    return [s for s in segments if s["segment_id"] in selected_ids]

def enforce_narrative_rules(selected: List[dict], full_transcript: List[dict]) -> List[dict]:
    fixed, processed = [], set()
    for seg in selected:
        if seg["segment_id"] in processed: continue
        fixed.append(seg)
        processed.add(seg["segment_id"])
        check = seg
        while check["text"].strip() and check["text"].strip()[-1] not in [".", "!", "?"]:
            next_id = check["segment_id"] + 1
            if next_id >= len(full_transcript): break
            next_seg = full_transcript[next_id]
            if next_seg["segment_id"] not in processed:
                fixed.append(next_seg)
                processed.add(next_seg["segment_id"])
            check = next_seg
    fixed.sort(key=lambda x: x["segment_id"])
    return fixed

# ==========================================
# 3. EXPORT ENGINES
# ==========================================
def export_audio_supercut(source_path: str, segments: List[dict], output_path: str):
    audio = AudioSegment.from_file(source_path)
    combined = AudioSegment.empty()
    inter_gap = AudioSegment.silent(duration=INTER_SEGMENT_GAP_MS, frame_rate=audio.frame_rate)
    big_gap = AudioSegment.silent(duration=GAP_MS, frame_rate=audio.frame_rate)

    for i, seg in enumerate(segments):
        # Small head padding only for the very first segment or big jumps
        start_pad = HEAD_PADDING_MS if i == 0 else 5
        
        start_ms = max(0, seg["start"] - start_pad)
        end_ms = seg["end"] + TAIL_PADDING_MS

        clip = audio[start_ms:end_ms]

        if i == 0:
            clip = clip.fade_in(FADE_MS)
        else:
            clip = clip.fade_in(18).fade_out(18)

        combined += clip

        # Add small silent gap between segments (this helps avoid overlap)
        if i < len(segments) - 1:
            if segments[i+1]["segment_id"] > seg["segment_id"] + 1:
                combined += big_gap      # bigger gap when skipping
            else:
                combined += inter_gap    # small gap between adjacent segments

    ext = Path(output_path).suffix.replace(".", "")
    combined.export(output_path, format=ext, bitrate=BITRATE)

def export_video_supercut(source_path: str, segments: List[dict], output_path: str):
    video = VideoFileClip(source_path)
    clips = []

    for i, seg in enumerate(segments):
        start_pad = HEAD_PADDING_MS if i == 0 else 5
        
        start_s = max(0, (seg["start"] - start_pad) / 1000.0)
        end_s = (seg["end"] + TAIL_PADDING_MS) / 1000.0

        clip = video.subclip(start_s, end_s)

        if i == 0:
            clip = clip.fadein(FADE_MS / 1000)
        else:
            clip = clip.fadein(0.018).fadeout(0.018)

        clips.append(clip)

    # For video, we use concatenate with method="compose" and let the small gap come from timing
    # But to add small gap between clips, we can insert short black/silent subclips if needed.
    # Simpler approach: just use the clips as-is with fade (small gap via timing is enough)

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    video.close()

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
def create_podcast_supercut(audio_path: str, target_percent: float) -> dict:
    try:
        # 1. Setup
        client = initialize_environment()
        working_audio, base_name, is_video = prepare_working_audio(audio_path)
        original_ext = Path(audio_path).suffix

        # 2. Transcription
        full_transcript = run_transcription(working_audio)
        full_json_path = save_json_artifact(full_transcript, DEFAULT_OUTPUT_DIR, f"{base_name}_full.json")
        
        # 3. Processing
        orig_sec = (full_transcript[-1]["end"] - full_transcript[0]["start"]) / 1000.0   # ← FIXED
        target_sec = orig_sec * (target_percent / 100)
        target_sec = orig_sec * (target_percent / 100)
        
        ai_selection = get_ai_selection(client, full_transcript, target_percent)
        final_segments = enforce_narrative_rules(ai_selection, full_transcript)
        edited_json_path = save_json_artifact(final_segments, DEFAULT_OUTPUT_DIR, f"{base_name}_edited.json")
        
        # 4. Export
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_name}_supercut{original_ext}")
        if is_video:
            export_video_supercut(audio_path, final_segments, final_path)
        else:
            export_audio_supercut(audio_path, final_segments, final_path)

        # 5. Cleanup
        if os.path.exists(working_audio): os.remove(working_audio)

        # 6. Detailed Return
        return {
            "status": "success",
            "original_duration_seconds": round(orig_sec, 1),
            "target_duration_seconds": round(target_sec, 1),
            "original_file_path": str(Path(audio_path).absolute()),
            "final_supercut_path": final_path,
            "edited_json_path": edited_json_path,
            "full_transcript_json": full_json_path,
            "number_of_segments_original": len(full_transcript),
            "number_of_segments_final": len(final_segments),
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    p = float(input("Enter percentage: "))
    path = r"app\files\demo\Tucker_ Yes_ Epstein Was Murdered and They Covered It Up.mp3" 
    print(json.dumps(create_podcast_supercut(path, p), indent=2))