"""Gemini Live API SocketIO event handlers."""

import asyncio
import base64
import os
import re
import tempfile
import threading

from flask import request
from flask_socketio import emit
from google import genai
from pydub import AudioSegment

from handlers.search import process_pending_search
from src.core.config import (
    GEMINI_INFERENCE_CONFIG,
    GEMINI_MODEL,
    GEMINI_SEND_SAMPLE_RATE,
    GEMINI_TRANSCRIPTION_CONFIG,
    GOOGLE_API_KEY,
)
from src.core.session import TranscriptionSession

# Prompt to send when user requests search query inference
INFER_SEARCH_PROMPT = (
    "Which exact term on the audio should I search for based on what you just heard?"
)

# Regex patterns for parsing Gemini inference response
# Allow variations: [SearchQuery], [ SearchQuery], [[SearchQuery], [  SearchQuery ], etc.
SEARCH_QUERY_PATTERN = re.compile(r"\[+\s*SearchQuery\s*\]", re.IGNORECASE)
# Pattern to extract Term and Definition (more flexible)
TERM_PATTERN = re.compile(r"Term:\s*([^\n\[]+?)(?:\s*Definition:|\s*Done\.|\s*\[|$)", re.IGNORECASE)
DEFINITION_PATTERN = re.compile(r"Definition:\s*([^\[\n]+?)(?:\s*Done\.|\s*\[|$)", re.IGNORECASE)
DONE_PATTERN = re.compile(r"Done\.?\s*$", re.IGNORECASE)

# Patterns for bad/unhelpful responses that should be filtered out
# Only filter clearly useless responses - be less strict to allow more through
BAD_RESPONSE_PATTERNS = [
    re.compile(r"^\s*\.?\s*Done\.?\s*$", re.IGNORECASE),  # Just "Done." or ". Done."
    re.compile(r"^\s*\[?\s*SearchQuery\s*\]?\s*I have already suggested.+Done\.?\s*$", re.IGNORECASE),  # "[SearchQuery] I have already suggested X. Done."
    re.compile(r"^\s*I have already suggested.+Done\.?\s*$", re.IGNORECASE),  # "I have already suggested X. Done." without [SearchQuery]
]


def is_bad_response(text):
    """Check if the response is a bad/unhelpful response that should be filtered.

    Only filters clearly useless responses. Returns False for most responses
    to allow them through (lower barrier).

    Returns:
        True if the response should be filtered out, False otherwise.
    """
    if not text:
        return True

    # Check against bad response patterns
    for pattern in BAD_RESPONSE_PATTERNS:
        if pattern.match(text.strip()):  # Use match instead of search for stricter checking
            return True

    return False


def parse_inference_response(text):
    """Parse Gemini inference response to extract search terms and definitions.

    Expected format (can have multiple):
    [SearchQuery]
    Term: <keyword>
    Definition: <explanation>
    [SearchQuery]
    Term: <keyword2>
    Definition: <explanation2>
    Done.

    Returns:
        dict with keys: terms (list), is_search_query, is_done
    """
    result = {
        "terms": [],  # List of {term, definition}
        "term": None,  # First term (for backwards compatibility)
        "definition": None,  # First definition (for backwards compatibility)
        "is_search_query": False,
        "is_done": False,
        "raw_text": text,  # Keep raw text for fallback
    }

    if not text:
        return result

    # Normalize text - remove extra brackets and spaces around SearchQuery
    normalized = re.sub(r'\[\s*\[+', '[', text)  # [[ -> [
    normalized = re.sub(r'\[\s+', '[', normalized)  # [ SearchQuery] -> [SearchQuery]

    # Check if this is a search query response (check both original and normalized)
    if SEARCH_QUERY_PATTERN.search(text) or SEARCH_QUERY_PATTERN.search(normalized):
        result["is_search_query"] = True

        # Split by [SearchQuery] markers to get individual blocks
        blocks = re.split(r'\[+\s*SearchQuery\s*\]', normalized, flags=re.IGNORECASE)

        for block in blocks:
            if not block.strip():
                continue

            # Extract term
            term_match = TERM_PATTERN.search(block)
            if term_match:
                term = term_match.group(1).strip()
                # Extract definition
                def_match = DEFINITION_PATTERN.search(block)
                definition = def_match.group(1).strip() if def_match else ""

                if term:
                    result["terms"].append({
                        "term": term,
                        "definition": definition
                    })

        # Set first term for backwards compatibility
        if result["terms"]:
            result["term"] = result["terms"][0]["term"]
            result["definition"] = result["terms"][0]["definition"]

    # Check if response is complete
    if DONE_PATTERN.search(text) or "Done." in text or "Done" in text:
        result["is_done"] = True

    return result


# Initialize Gemini client
gemini_client = None
if GOOGLE_API_KEY:
    gemini_client = genai.Client(
        http_options={"api_version": "v1beta"}, api_key=GOOGLE_API_KEY
    )
    print("[Gemini] Client initialized")
else:
    print("[Gemini] Warning: GOOGLE_API_KEY not set, Gemini Live will not work")

# Store Gemini Live sessions
gemini_live_sessions = {}


def run_gemini_live_loop(
    session_id, socketio, transcription_sessions, mode="transcription"
):
    """Background thread that runs the Gemini Live async event loop.

    Args:
        mode: "transcription" for speech-to-text, "inference" for search query inference
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def gemini_session_handler():
        """Main async handler for Gemini Live session."""
        if not gemini_client:
            socketio.emit(
                "error",
                {"message": "Gemini client not initialized. Set GOOGLE_API_KEY."},
                room=session_id,
            )
            return

        session = transcription_sessions.get(session_id)
        if not session:
            return

        # Select config based on mode
        if mode == "inference":
            config = GEMINI_INFERENCE_CONFIG
            print(f"[Gemini] Using INFERENCE mode for session {session_id}")
        else:
            config = GEMINI_TRANSCRIPTION_CONFIG
            print(f"[Gemini] Using TRANSCRIPTION mode for session {session_id}")

        try:
            print(f"[Gemini] Attempting to connect for session {session_id}...")
            socketio.emit(
                "status",
                {"message": f"Connecting to Gemini Live ({mode})..."},
                room=session_id,
            )

            async with gemini_client.aio.live.connect(
                model=GEMINI_MODEL, config=config
            ) as gemini_session:
                print(f"[Gemini] Connected successfully for session {session_id}")
                session.gemini_session = gemini_session
                session.gemini_active = True
                session.gemini_mode = mode  # Store the mode
                session.gemini_audio_queue = asyncio.Queue(maxsize=10)
                session.gemini_text_queue = asyncio.Queue(
                    maxsize=5
                )  # For inference requests

                socketio.emit(
                    "status",
                    {"message": "Gemini Live connected. Start speaking or play audio."},
                    room=session_id,
                )
                socketio.emit(
                    "gemini_connected", {"status": "connected"}, room=session_id
                )

                async def send_audio():
                    """Send audio chunks from queue to Gemini."""
                    while session.gemini_active:
                        try:
                            audio_data = await asyncio.wait_for(
                                session.gemini_audio_queue.get(), timeout=1.0
                            )
                            await gemini_session.send(input=audio_data)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            print(f"[Gemini] Send error: {e}")
                            break

                async def send_text_prompts():
                    """Send text prompts (for inference mode) to Gemini."""
                    while session.gemini_active:
                        try:
                            text_prompt = await asyncio.wait_for(
                                session.gemini_text_queue.get(), timeout=1.0
                            )
                            print(f"[Gemini] Sending text prompt: {text_prompt}")
                            await gemini_session.send(
                                input=text_prompt, end_of_turn=True
                            )
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            print(f"[Gemini] Text send error: {e}")
                            break

                async def receive_responses():
                    """Receive and process responses from Gemini."""
                    while session.gemini_active:
                        try:
                            turn = gemini_session.receive()
                            async for response in turn:
                                text = None

                                # Method 1: direct .text attribute
                                text = getattr(response, "text", None)

                                # Method 2: check for parts
                                if not text:
                                    parts = getattr(response, "parts", None)
                                    if parts:
                                        for part in parts:
                                            part_text = getattr(part, "text", None)
                                            if part_text:
                                                text = (text or "") + part_text

                                # Method 3: check for content
                                if not text:
                                    content = getattr(response, "content", None)
                                    if content:
                                        text = str(content)

                                # Method 4: check server_content (Live API specific)
                                if not text:
                                    server_content = getattr(
                                        response, "server_content", None
                                    )
                                    if server_content:
                                        model_turn = getattr(
                                            server_content, "model_turn", None
                                        )
                                        if model_turn:
                                            parts = getattr(model_turn, "parts", None)
                                            if parts:
                                                for part in parts:
                                                    part_text = getattr(
                                                        part, "text", None
                                                    )
                                                    if part_text:
                                                        text = (text or "") + part_text

                                if text:
                                    text = text.strip()
                                    if text:
                                        # Handle based on mode
                                        if mode == "inference":
                                            # Inference mode: buffer response until Done signal
                                            print(
                                                f"[Gemini Inference] Chunk received: {text}"
                                            )

                                            # Initialize buffer if not exists
                                            if not hasattr(session, "inference_buffer"):
                                                session.inference_buffer = ""

                                            # Append to buffer (replace newline with space)
                                            session.inference_buffer += text.replace("\n", " ") + " "

                                            # Emit streaming update to frontend (for live display)
                                            socketio.emit(
                                                "gemini_inference",
                                                {
                                                    "text": text,
                                                    "is_done": False,
                                                },
                                                room=session_id,
                                            )

                                            # Check if Done signal received
                                            if (
                                                DONE_PATTERN.search(text)
                                                or "Done." in text
                                            ):
                                                # Parse the complete buffered response
                                                full_response = (
                                                    session.inference_buffer.strip()
                                                )
                                                print(
                                                    f"[Gemini Inference] Done signal received. Full response:\n{full_response}"
                                                )

                                                # Check if this is a bad response that should be filtered
                                                if is_bad_response(full_response):
                                                    print(
                                                        f"[Gemini Inference] BAD RESPONSE FILTERED: {full_response}"
                                                    )
                                                    # Log the filtered response
                                                    session._log_event(
                                                        "gemini_inference_filtered",
                                                        {
                                                            "full_response": full_response,
                                                            "reason": "bad_response_pattern",
                                                        },
                                                    )
                                                    # Clear buffer and emit error to frontend
                                                    session.inference_buffer = ""
                                                    socketio.emit(
                                                        "gemini_inference",
                                                        {
                                                            "text": "",
                                                            "is_done": True,
                                                        },
                                                        room=session_id,
                                                    )
                                                    socketio.emit(
                                                        "gemini_search_terms",
                                                        {
                                                            "terms": [],
                                                            "total": 0,
                                                            "is_done": True,
                                                            "error": "No new terms available. Try again.",
                                                            "filtered": True,
                                                        },
                                                        room=session_id,
                                                    )
                                                    continue  # Skip to next response

                                                parsed = parse_inference_response(
                                                    full_response
                                                )
                                                print(
                                                    f"[Gemini Inference] Parsed result: {parsed}"
                                                )

                                                # Log the complete inference response
                                                session._log_event(
                                                    "gemini_inference_complete",
                                                    {
                                                        "full_response": full_response,
                                                        "parsed": parsed,
                                                    },
                                                )

                                                # Emit done signal
                                                socketio.emit(
                                                    "gemini_inference",
                                                    {
                                                        "text": "",
                                                        "is_done": True,
                                                    },
                                                    room=session_id,
                                                )

                                                # Always emit search terms event (even if parsing failed)
                                                # so frontend knows response is complete
                                                terms_to_send = parsed["terms"]

                                                # If no terms parsed, create fallback from raw text
                                                if not terms_to_send and parsed.get("raw_text"):
                                                    # Try to extract something useful
                                                    raw = parsed["raw_text"]
                                                    # Use first line or first 50 chars as term
                                                    fallback_term = raw.split('\n')[0][:50].strip()
                                                    if fallback_term:
                                                        terms_to_send = [{
                                                            "term": fallback_term,
                                                            "definition": "(parsing failed)"
                                                        }]

                                                if terms_to_send:
                                                    # Store all terms for navigation
                                                    session.gemini_recent_terms = terms_to_send
                                                    session.gemini_recent_term_index = 0

                                                    # Emit all extracted terms
                                                    socketio.emit(
                                                        "gemini_search_terms",
                                                        {
                                                            "terms": terms_to_send,
                                                            "total": len(terms_to_send),
                                                            "is_done": True,
                                                        },
                                                        room=session_id,
                                                    )

                                                    print(
                                                        f"[Gemini Inference] Extracted {len(terms_to_send)} terms: {[t['term'] for t in terms_to_send]}"
                                                    )
                                                else:
                                                    # No terms at all - still emit empty event
                                                    socketio.emit(
                                                        "gemini_search_terms",
                                                        {
                                                            "terms": [],
                                                            "total": 0,
                                                            "is_done": True,
                                                            "error": "No terms could be parsed",
                                                        },
                                                        room=session_id,
                                                    )
                                                    print("[Gemini Inference] No terms could be parsed")

                                                # Clear buffer for next inference request
                                                session.inference_buffer = ""
                                        else:
                                            # Transcription mode: standard handling
                                            print(f"[Gemini] Transcription: {text}")

                                            # Add to session (for GPT keyword extraction)
                                            session.add_text(text)

                                            # Emit as standard transcription (like Whisper)
                                            socketio.emit(
                                                "transcription",
                                                {
                                                    "text": text,
                                                    "source": "gemini-live",
                                                },
                                                room=session_id,
                                            )

                                            # Log transcription
                                            session._log_event(
                                                "gemini_transcription",
                                                {
                                                    "text": text,
                                                    "word_count": len(text.split()),
                                                },
                                            )

                                            # Check for pending search and process it
                                            if session.pending_search:
                                                print(
                                                    "[Gemini] Pending search detected, processing..."
                                                )
                                                process_pending_search(
                                                    session, session_id, socketio
                                                )
                                else:
                                    data = getattr(response, "data", None)
                                    if data:
                                        print(
                                            f"[Gemini DEBUG] Got audio data: {len(data)} bytes"
                                        )

                        except Exception as e:
                            if session.gemini_active:
                                print(f"[Gemini] Receive error: {e}")
                                import traceback

                                traceback.print_exc()
                            break

                send_task = asyncio.create_task(send_audio())
                send_text_task = asyncio.create_task(send_text_prompts())
                receive_task = asyncio.create_task(receive_responses())

                while session.gemini_active:
                    await asyncio.sleep(0.5)

                send_text_task.cancel()

                send_task.cancel()
                receive_task.cancel()

        except Exception as e:
            print(f"[Gemini] Session error: {e}")
            import traceback

            traceback.print_exc()
            socketio.emit(
                "error", {"message": f"Gemini error: {str(e)}"}, room=session_id
            )
        finally:
            session.gemini_active = False
            session.gemini_session = None
            socketio.emit(
                "gemini_disconnected", {"status": "disconnected"}, room=session_id
            )

    try:
        loop.run_until_complete(gemini_session_handler())
    finally:
        loop.close()


def register_gemini_handlers(socketio, transcription_sessions):
    """Register Gemini-related event handlers."""

    @socketio.on("start_gemini_live")
    def handle_start_gemini_live(data=None):
        """Start Gemini Live session for real-time transcription or inference."""
        session_id = request.sid
        mode = data.get("mode", "transcription") if data else "transcription"

        if not gemini_client:
            emit(
                "error",
                {"message": "Gemini client not initialized. Set GOOGLE_API_KEY."},
            )
            return

        if session_id not in transcription_sessions:
            transcription_sessions[session_id] = TranscriptionSession(session_id)

        session = transcription_sessions[session_id]

        if session.gemini_active:
            emit("status", {"message": "Gemini Live already active"})
            return

        # Reset Gemini state
        session.gemini_raw_output = ""
        session.gemini_parse_buffer = ""  # Reset buffer for new session
        session.gemini_captions = ""
        session.gemini_summary = {"overall_context": "", "current_segment": ""}
        session.gemini_terms = []
        session.gemini_recent_terms = []  # Reset recent terms
        session.gemini_recent_term_index = 0
        session.inference_buffer = ""  # Reset inference buffer

        session._log_event("gemini_live_start", {"model": GEMINI_MODEL, "mode": mode})

        emit("status", {"message": f"Starting Gemini Live ({mode})..."})

        thread = threading.Thread(
            target=run_gemini_live_loop,
            args=(session_id, socketio, transcription_sessions, mode),
        )
        thread.daemon = True
        thread.start()
        gemini_live_sessions[session_id] = thread

    @socketio.on("gemini_infer_search")
    def handle_gemini_infer_search():
        """Request Gemini Live to infer search query based on what it heard."""
        session_id = request.sid

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]

        if not session.gemini_active or not session.gemini_text_queue:
            emit(
                "error", {"message": "Gemini Live not active. Start Gemini Live first."}
            )
            return

        # Send the inference prompt
        try:
            session.gemini_text_queue.put_nowait(INFER_SEARCH_PROMPT)
            print("[Gemini] Sent inference request")
            session._log_event("gemini_infer_request", {})
            emit("status", {"message": "Asking Gemini for search suggestion..."})
        except Exception as e:
            print(f"[Gemini] Failed to send inference request: {e}")
            emit("error", {"message": "Failed to send inference request."})

    @socketio.on("stop_gemini_live")
    def handle_stop_gemini_live():
        """Stop Gemini Live session."""
        session_id = request.sid

        if session_id in transcription_sessions:
            session = transcription_sessions[session_id]
            session.gemini_active = False

            session._log_event(
                "gemini_live_stop",
                {
                    "total_terms": len(session.gemini_terms_history),
                    "final_captions_length": len(session.gemini_captions),
                },
            )

        emit("status", {"message": "Gemini Live session stopped"})
        emit("gemini_disconnected", {"status": "disconnected"})

    @socketio.on("audio_chunk_gemini_live")
    def handle_audio_chunk_gemini_live(data):
        """Process audio chunk with Gemini Live."""
        session_id = request.sid

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]

        if not session.gemini_active or not session.gemini_audio_queue:
            emit("error", {"message": "Gemini Live not active. Start Gemini first."})
            return

        try:
            audio_data = base64.b64decode(data["audio"])
            file_format = data.get("format", "webm")

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=f".{file_format}", delete=False
                ) as temp_audio:
                    temp_audio.write(audio_data)
                    temp_path = temp_audio.name

                audio = AudioSegment.from_file(temp_path, format=file_format)
                audio = (
                    audio.set_channels(1)
                    .set_frame_rate(GEMINI_SEND_SAMPLE_RATE)
                    .set_sample_width(2)
                )
                raw_data = audio.raw_data

                audio_input = {"data": raw_data, "mime_type": "audio/pcm"}

                def add_to_queue():
                    try:
                        session.gemini_audio_queue.put_nowait(audio_input)
                    except asyncio.QueueFull:
                        print("[Gemini] Audio queue full, dropping chunk")

                if session.gemini_audio_queue:
                    add_to_queue()

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            print(f"[Gemini] Audio chunk error: {e}")
            import traceback

            traceback.print_exc()
            emit("error", {"message": f"Gemini audio error: {str(e)}"})
