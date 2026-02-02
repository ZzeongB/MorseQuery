"""Search-related SocketIO event handlers."""

import threading
from datetime import datetime

from flask import request
from flask_socketio import emit

from src.core.search import google_custom_search


def _fetch_image_for_keyword(keyword: str) -> str | None:
    """Fetch a representative image for a keyword using Google Image Search."""
    try:
        results = google_custom_search(keyword, search_type="image")
        if results and len(results) > 0:
            # Return the first image's link (full image URL)
            return results[0].get("link") or results[0].get("thumbnail")
    except Exception as e:
        print(f"[Image Search] Error fetching image for '{keyword}': {e}")
    return None


def _start_pending_search_timeout(session, session_id, socketio, timeout_seconds=10.0):
    """Start a background thread to process pending search after timeout."""

    def timeout_handler():
        import time

        time.sleep(timeout_seconds)

        if session.pending_search:
            print(
                f"[Pending Search] Timeout after {timeout_seconds}s, processing with current text"
            )
            process_pending_search(session, session_id, socketio)

    thread = threading.Thread(target=timeout_handler)
    thread.daemon = True
    thread.start()


def process_pending_search(session, session_id, socketio):
    """Process a pending search after new transcription arrives."""
    if not session.pending_search:
        return

    pending = session.pending_search
    session.pending_search = None

    time_threshold = pending.get("time_threshold", 15)

    print(f"[Pending Search] Processing with {len(session.words)} words available")

    # Log the delay
    spacebar_time = pending.get("timestamp")
    if spacebar_time:
        delay = (datetime.utcnow() - spacebar_time).total_seconds()
        print(f"[Pending Search] Delay from spacebar: {delay:.2f}s")
        session._log_event(
            "pending_search_processed",
            {
                "delay_seconds": delay,
                "words_before": pending.get("words_before", 0),
                "words_after": len(session.words),
                "new_text_received": len(session.words)
                - pending.get("words_before", 0),
            },
        )

    # Call GPT - now returns list of keyword pairs
    keywords = session.get_top_keyword_gpt(time_threshold)

    if not keywords:
        socketio.emit(
            "error", {"message": "No keywords available for search"}, room=session_id
        )
        return

    # Fetch images for keywords
    for kw in keywords:
        kw["image"] = _fetch_image_for_keyword(kw.get("keyword", ""))

    # Emit the keyword list (single press result)
    socketio.emit(
        "keywords_extracted",
        {
            "keywords": keywords,
            "history": session.keyword_history,
        },
        room=session_id,
    )

    session.log_search_action(
        search_mode="gpt",
        search_type="text",
        keyword=keywords[0]["keyword"] if keywords else None,
        num_results=len(keywords),
    )


def register_search_handlers(socketio, transcription_sessions):
    """Register search-related event handlers."""

    @socketio.on("search_request")
    def handle_search_request(data):
        """Handle search request triggered by single spacebar press.

        Returns list of keywords.
        """
        session_id = request.sid
        client_timestamp = data.get("client_timestamp")
        time_threshold = 15

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]

        # Log action
        session.log_search_action(search_mode="gpt", search_type="text")

        if client_timestamp:
            try:
                client_time = datetime.fromisoformat(
                    client_timestamp.replace("Z", "+00:00")
                )
                latency = (datetime.utcnow() - client_time).total_seconds()
                print(f"[TIMING] Client pressed spacebar at: {client_timestamp}")
                print(f"[TIMING] Client-server latency: {latency:.3f}s")
            except Exception as e:
                print(f"[TIMING] Could not parse client timestamp: {e}")

        # If OpenAI Realtime is active, wait for next transcription before calling GPT
        if session.openai_active:
            print(
                f"[GPT Search] OpenAI active, setting pending search (words so far: {len(session.words)})"
            )
            session.pending_search = {
                "timestamp": datetime.utcnow(),
                "time_threshold": time_threshold,
                "words_before": len(session.words),
                "client_timestamp": client_timestamp,
                "source": "openai",
            }
            session._log_event(
                "pending_search_set",
                {
                    "words_at_spacebar": len(session.words),
                    "client_timestamp": client_timestamp,
                    "source": "openai",
                },
            )
            emit("status", {"message": "Extracting keywords..."})
            _start_pending_search_timeout(
                session, session_id, socketio, timeout_seconds=10.0
            )
            return

        # If not streaming, process immediately
        keywords = session.get_top_keyword_gpt(time_threshold)

        if not keywords:
            emit("error", {"message": "No keywords available for search"})
            return

        # Fetch images for keywords
        for kw in keywords:
            kw["image"] = _fetch_image_for_keyword(kw.get("keyword", ""))

        # Emit the keyword list
        emit(
            "keywords_extracted",
            {
                "keywords": keywords,
                "history": session.keyword_history,
            },
        )

        session.log_search_action(
            search_mode="gpt",
            search_type="text",
            keyword=keywords[0]["keyword"] if keywords else None,
            num_results=len(keywords),
        )

    @socketio.on("select_keyword")
    def handle_select_keyword(data):
        """Handle double-spacebar or click to select a keyword and show its description."""
        session_id = request.sid
        keyword_index = data.get("index", 0)

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]

        if not session.gpt_keyword_pairs:
            emit("error", {"message": "No keywords available. Press spacebar first."})
            return

        # Validate index
        if keyword_index < 0 or keyword_index >= len(session.gpt_keyword_pairs):
            keyword_index = 0

        session.current_keyword_index = keyword_index
        current_pair = session.gpt_keyword_pairs[keyword_index]

        emit(
            "keyword_selected",
            {
                "keyword": current_pair.get("keyword", ""),
                "description": current_pair.get("description", ""),
                "index": keyword_index,
                "total": len(session.gpt_keyword_pairs),
            },
        )

    @socketio.on("next_keyword")
    def handle_next_keyword(data):
        """Handle double-spacebar to navigate to next keyword."""
        session_id = request.sid

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]

        if not session.gpt_keyword_pairs:
            emit("error", {"message": "No keywords available. Press spacebar first."})
            return

        # Move to next keyword
        session.current_keyword_index += 1
        if session.current_keyword_index >= len(session.gpt_keyword_pairs):
            session.current_keyword_index = 0

        current_pair = session.gpt_keyword_pairs[session.current_keyword_index]

        emit(
            "keyword_selected",
            {
                "keyword": current_pair.get("keyword", ""),
                "description": current_pair.get("description", ""),
                "index": session.current_keyword_index,
                "total": len(session.gpt_keyword_pairs),
            },
        )

    @socketio.on("search_grounding")
    def handle_search_grounding(data):
        """Handle long-press: Google Search Grounding using Gemini API."""
        import os

        from google import genai
        from google.genai import types

        session_id = request.sid
        keyword = data.get("keyword", "")
        description = data.get("description", "")
        if not keyword:
            emit("error", {"message": "No keyword provided for grounding search"})
            return

        print(f"[Search Grounding] Keyword: {keyword}, Description: {description}")

        # Get prompt config from session
        session = transcription_sessions.get(session_id)
        prompt_prefix = ""
        if session:
            session_config = session.get_current_config()
            prompt_prefix = session_config.get("gemini_search_prompt_prefix", "")
            if prompt_prefix:
                print(f"[Search Grounding] Using prompt prefix: {prompt_prefix}")

        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                emit("error", {"message": "GOOGLE_API_KEY not configured"})
                return

            client = genai.Client(api_key=api_key)
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[grounding_tool])

            if description:
                prompt = (
                    f"{prompt_prefix}"
                    f"Please provide detailed information in 1–2 paragraphs about: "
                    f"{keyword} ({description}). "
                    f"Highlight only the most important terms and concepts using **bold**, "
                    f"and do not overuse bold formatting."
                )
            else:
                prompt = (
                    f"{prompt_prefix}"
                    f"Please provide detailed information in 1–2 paragraphs about: "
                    f"{keyword}. "
                    f"Highlight only the most important terms and concepts using **bold**, "
                    f"and do not overuse bold formatting."
                )

            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=config,
            )

            text = response.text
            citations = []

            if (
                response.candidates
                and response.candidates[0].grounding_metadata
                and response.candidates[0].grounding_metadata.grounding_supports
            ):
                supports = response.candidates[0].grounding_metadata.grounding_supports
                chunks = response.candidates[0].grounding_metadata.grounding_chunks

                sorted_supports = sorted(
                    supports, key=lambda s: s.segment.end_index, reverse=True
                )

                for support in sorted_supports:
                    end_index = support.segment.end_index
                    if support.grounding_chunk_indices:
                        citation_refs = []
                        for i in support.grounding_chunk_indices:
                            if i < len(chunks):
                                uri = chunks[i].web.uri
                                title = getattr(
                                    chunks[i].web, "title", f"Source {i + 1}"
                                )
                                existing = next(
                                    (c for c in citations if c["uri"] == uri), None
                                )
                                if not existing:
                                    citations.append(
                                        {
                                            "index": len(citations) + 1,
                                            "uri": uri,
                                            "title": title,
                                        }
                                    )
                                    citation_refs.append(len(citations))
                                else:
                                    citation_refs.append(existing["index"])

                        citation_string = "".join([f"[{ref}]" for ref in citation_refs])
                        text = text[:end_index] + citation_string + text[end_index:]

            print(f"[Search Grounding] Response: {text[:100]}...")
            print(f"[Search Grounding] Citations: {len(citations)}")

            # Fetch image for the keyword
            image_url = _fetch_image_for_keyword(keyword)

            emit(
                "grounding_result",
                {
                    "keyword": keyword,
                    "text": text,
                    "citations": citations,
                    "image": image_url,
                },
            )

        except Exception as e:
            print(f"Search grounding error: {str(e)}")
            emit("error", {"message": f"Search grounding error: {str(e)}"})

    @socketio.on("get_history")
    def handle_get_history():
        """Get keyword history."""
        session_id = request.sid

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]
        emit("keyword_history", {"history": session.keyword_history})
