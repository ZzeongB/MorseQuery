"""Gemini direct audio inference SocketIO event handlers.

This module handles direct audio-to-keyword inference using Gemini 3.0 Flash,
bypassing the transcription step entirely.
"""

import base64
import os
import struct
import tempfile
from datetime import datetime
from typing import Dict, List

from flask import request
from flask_socketio import emit

from src.core.search import google_custom_search

# Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-3.0-flash-preview"

# Audio buffer settings
MAX_AUDIO_BUFFER_SECONDS = 30  # Maximum audio to keep in buffer
SAMPLE_RATE = 24000  # 24kHz


def _fetch_image_for_keyword(keyword: str) -> str | None:
    """Fetch a representative image for a keyword using Google Image Search."""
    try:
        results = google_custom_search(keyword, search_type="image")
        if results and len(results) > 0:
            return results[0].get("link") or results[0].get("thumbnail")
    except Exception as e:
        print(f"[Image Search] Error fetching image for '{keyword}': {e}")
    return None


def extract_keywords_from_audio_gemini(
    audio_data: bytes, session, auto_mode: bool = False
) -> List[Dict]:
    """Extract keywords directly from audio using Gemini 3.0 Flash.

    Args:
        audio_data: Raw PCM audio bytes (16-bit, mono, 24kHz)
        session: TranscriptionSession for history tracking
        auto_mode: If True, allow 0-3 keywords; if False, exactly 3

    Returns:
        List of keyword-description pairs
    """
    from google import genai
    from google.genai import types

    if not GOOGLE_API_KEY:
        print("[Gemini Audio] GOOGLE_API_KEY not configured")
        return []

    if not audio_data or len(audio_data) < 1000:
        print("[Gemini Audio] Audio data too small")
        return []

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)

        # Build exclusion list from history
        history_keywords = session.get_history_keywords_list() if session else []
        exclusion_text = ""
        if history_keywords:
            exclusion_text = f"\n\nIMPORTANT: Do NOT suggest these keywords that were already extracted: {', '.join(history_keywords)}"
            print(f"[Gemini Audio] Excluding keywords: {history_keywords}")

        # Create prompt for keyword extraction
        if auto_mode:
            prompt = f"""Listen to this audio and extract important keywords that users might want to look up.

Identify words or phrases worth looking up. The selected words or phrases should be:
- Technical terms or unfamiliar vocabulary
- Concepts that need clarification
- Names or specific references
- Words that might need visual aids

Return 0 to 3 keywords. If there are no important keywords, respond with "None".
If there are keywords, use this format for each:
Keyword: <word or phrase>
Description: <a brief 1-sentence description>{exclusion_text}"""
        else:
            prompt = f"""Listen to this audio and predict the top three words or phrases the user would most likely want to look up.

The selected words or phrases should be:
- Technical terms or unfamiliar vocabulary
- Concepts that need clarification
- Names or specific references
- Words that might need visual aids

Respond with EXACTLY 3 keyword-description pairs in this format:
Keyword: <word or phrase 1 - most important>
Description: <a brief 1-sentence description>
Keyword: <word or phrase 2 - second most important>
Description: <a brief 1-sentence description>
Keyword: <word or phrase 3 - third most important>
Description: <a brief 1-sentence description>{exclusion_text}"""

        # Create WAV file from PCM data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            num_samples = len(audio_data) // 2
            wav_header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                36 + len(audio_data),
                b"WAVE",
                b"fmt ",
                16,  # Subchunk1Size
                1,  # AudioFormat (PCM)
                1,  # NumChannels (mono)
                SAMPLE_RATE,
                SAMPLE_RATE * 2,  # ByteRate
                2,  # BlockAlign
                16,  # BitsPerSample
                b"data",
                len(audio_data),
            )
            f.write(wav_header)
            f.write(audio_data)
            temp_audio_path = f.name

        try:
            # Upload audio file to Gemini
            audio_file = client.files.upload(file=temp_audio_path)
            print(f"[Gemini Audio] Uploaded audio: {audio_file.name}, size: {len(audio_data)} bytes")

            # Generate content with audio
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_uri(
                                file_uri=audio_file.uri, mime_type="audio/wav"
                            ),
                            types.Part.from_text(prompt),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                ),
            )

            raw_response = response.text.strip()
            print(f"[Gemini Audio] Response: {raw_response}")

            # Parse response
            keyword_description_pairs = []
            current_keyword = None

            for line in raw_response.split("\n"):
                line = line.strip()
                if line.lower().startswith("keyword:"):
                    current_keyword = line.split(":", 1)[1].strip()
                elif line.lower().startswith("description:") and current_keyword:
                    description = line.split(":", 1)[1].strip()
                    if current_keyword.lower() not in [
                        h.lower() for h in history_keywords
                    ]:
                        keyword_description_pairs.append(
                            {
                                "keyword": current_keyword,
                                "description": description,
                            }
                        )
                    current_keyword = None

            if current_keyword and current_keyword.lower() not in [
                h.lower() for h in history_keywords
            ]:
                keyword_description_pairs.append(
                    {
                        "keyword": current_keyword,
                        "description": None,
                    }
                )

            # Update session state
            if session:
                session.gpt_keyword_pairs = keyword_description_pairs
                session.current_keyword_index = 0
                session.add_keywords_to_history(keyword_description_pairs)

                session._log_event(
                    "keyword_extraction_gemini_audio",
                    {
                        "audio_size_bytes": len(audio_data),
                        "raw_response": raw_response,
                        "keyword_description_pairs": keyword_description_pairs,
                        "excluded_keywords": history_keywords,
                        "model": GEMINI_MODEL,
                    },
                )

            return keyword_description_pairs

        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    except Exception as e:
        print(f"[Gemini Audio] Error: {e}")
        import traceback

        traceback.print_exc()
        if session:
            session._log_event(
                "keyword_extraction_gemini_audio",
                {
                    "audio_size_bytes": len(audio_data) if audio_data else 0,
                    "error": str(e),
                },
            )
        return []


def register_gemini_inference_handlers(
    socketio, transcription_sessions, get_inference_mode
):
    """Register Gemini direct inference event handlers.

    Args:
        socketio: Flask-SocketIO instance
        transcription_sessions: Dict to store session data
        get_inference_mode: Function that returns current inference mode
    """

    @socketio.on("audio_chunk_gemini")
    def handle_audio_chunk_gemini(data):
        """Process audio chunk for Gemini direct inference mode.

        Accumulates audio in session buffer for later inference.
        """
        session_id = request.sid
        session = transcription_sessions.get(session_id)

        if not session:
            emit("error", {"message": "No active session"})
            return

        if get_inference_mode() != "direct_audio":
            return

        try:
            audio_data = base64.b64decode(data["audio"])
            file_format = data.get("format", "webm")

            # Initialize audio buffer if needed
            if not hasattr(session, "gemini_audio_buffer"):
                session.gemini_audio_buffer = bytearray()
                session.gemini_audio_start_time = datetime.utcnow()

            # Convert to PCM using existing function
            from handlers.openai_transcription import convert_webm_to_pcm

            pcm_float = convert_webm_to_pcm(audio_data, file_format)

            # Convert float32 to int16 PCM bytes
            pcm_bytes = b"".join(
                struct.pack("<h", int(max(-1.0, min(1.0, x)) * 32767)) for x in pcm_float
            )

            session.gemini_audio_buffer.extend(pcm_bytes)

            # Trim buffer if too long
            max_bytes = MAX_AUDIO_BUFFER_SECONDS * SAMPLE_RATE * 2
            if len(session.gemini_audio_buffer) > max_bytes:
                session.gemini_audio_buffer = session.gemini_audio_buffer[-max_bytes:]

        except Exception as e:
            print(f"[Gemini Audio] Audio processing error: {e}")

    @socketio.on("search_request_gemini_audio")
    def handle_search_request_gemini_audio(data):
        """Handle search request for Gemini direct audio inference."""
        session_id = request.sid
        session = transcription_sessions.get(session_id)

        if not session:
            emit("error", {"message": "No active session"})
            return

        if (
            not hasattr(session, "gemini_audio_buffer")
            or len(session.gemini_audio_buffer) < 1000
        ):
            emit("error", {"message": "Not enough audio. Please speak first."})
            return

        emit("status", {"message": "Extracting keywords from audio..."})

        keywords = extract_keywords_from_audio_gemini(
            bytes(session.gemini_audio_buffer), session, auto_mode=False
        )

        if not keywords:
            emit("error", {"message": "No keywords extracted from audio"})
            return

        for kw in keywords:
            kw["image"] = _fetch_image_for_keyword(kw.get("keyword", ""))

        emit(
            "keywords_extracted",
            {
                "keywords": keywords,
                "history": session.keyword_history,
                "mode": "gemini_audio",
            },
        )

        session.log_search_action(
            search_mode="gemini_audio",
            search_type="text",
            keyword=keywords[0]["keyword"] if keywords else None,
            num_results=len(keywords),
        )

    @socketio.on("clear_gemini_buffer")
    def handle_clear_gemini_buffer():
        """Clear the Gemini audio buffer."""
        session_id = request.sid
        session = transcription_sessions.get(session_id)

        if session and hasattr(session, "gemini_audio_buffer"):
            session.gemini_audio_buffer = bytearray()
            session.gemini_audio_start_time = datetime.utcnow()
            emit("status", {"message": "Audio buffer cleared"})
