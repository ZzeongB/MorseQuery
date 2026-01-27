"""Search-related SocketIO event handlers."""

import threading
from datetime import datetime

from flask import request
from flask_socketio import emit

from src.core.search import count_results, google_custom_search


def _start_pending_search_timeout(session, session_id, socketio, timeout_seconds=2.0):
    """Start a background thread to process pending search after timeout.

    If the pending search is still present after timeout, process it with current text.
    """
    def timeout_handler():
        import time
        time.sleep(timeout_seconds)

        # Check if pending search still exists (not yet processed by transcription)
        if session.pending_search:
            print(f"[Pending Search] Timeout after {timeout_seconds}s, processing with current text")
            process_pending_search(session, session_id, socketio)

    thread = threading.Thread(target=timeout_handler)
    thread.daemon = True
    thread.start()


def process_pending_search(session, session_id, socketio):
    """Process a pending search after new transcription arrives.

    Called from gemini.py when new transcription is received.
    """
    if not session.pending_search:
        return

    pending = session.pending_search
    session.pending_search = None  # Clear pending state

    search_type = pending.get("search_type", "text")
    skip_search = pending.get("skip_search", False)
    show_all_keywords = pending.get("show_all_keywords", False)
    time_threshold = pending.get("time_threshold", 10)

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
                "new_text_received": len(session.words) - pending.get("words_before", 0),
            },
        )

    # Now call GPT with the updated text
    keyword = session.get_top_keyword_gpt(time_threshold)

    if not keyword:
        socketio.emit("error", {"message": "No keywords available for search"}, room=session_id)
        return

    keyword_clean = "".join(c for c in keyword if c.isalnum() or c.isspace()).strip()

    description = None
    total_keywords = 1
    current_index = 0

    if session.gpt_keyword_pairs:
        total_keywords = len(session.gpt_keyword_pairs)
        current_index = session.current_keyword_index
        if current_index < len(session.gpt_keyword_pairs):
            description = session.gpt_keyword_pairs[current_index].get("description")

    if show_all_keywords and session.gpt_keyword_pairs:
        socketio.emit(
            "all_keywords",
            {"keywords": session.gpt_keyword_pairs, "mode": "gpt"},
            room=session_id,
        )
        if skip_search:
            return
    else:
        socketio.emit(
            "search_keyword",
            {
                "keyword": keyword_clean,
                "mode": "gpt",
                "description": description,
                "total_keywords": total_keywords,
                "current_index": current_index,
            },
            room=session_id,
        )

    if skip_search:
        return

    try:
        search_results = google_custom_search(keyword_clean, search_type)
        num_results = count_results(search_results, search_type)

        session.log_search_action(
            search_mode="gpt_delayed",
            search_type=search_type,
            keyword=keyword_clean,
            num_results=num_results,
        )

        socketio.emit(
            "search_results",
            {
                "keyword": keyword_clean,
                "mode": "gpt",
                "type": search_type,
                "results": search_results,
            },
            room=session_id,
        )
    except Exception as e:
        print(f"Search error: {str(e)}")
        socketio.emit("error", {"message": f"Search error: {str(e)}"}, room=session_id)


def register_search_handlers(socketio, transcription_sessions):
    """Register search-related event handlers."""

    @socketio.on("search_request")
    def handle_search_request(data):
        """Handle search request triggered by spacebar."""
        session_id = request.sid
        search_request_time = datetime.utcnow()

        search_mode = data.get("mode", "gpt")
        search_type = data.get("type", "text")
        client_timestamp = data.get("client_timestamp")
        skip_search = data.get("skip_search", False)
        show_all_keywords = data.get("show_all_keywords", False)

        # Log action
        transcription_sessions[session_id].log_search_action(
            search_mode=search_mode,
            search_type=search_type,
        )

        if client_timestamp:
            try:
                client_time = datetime.fromisoformat(
                    client_timestamp.replace("Z", "+00:00")
                )
                latency = (search_request_time - client_time).total_seconds()
                print(f"[TIMING] Client pressed spacebar at: {client_timestamp}")
                print(f"[TIMING] Client-server latency: {latency:.3f}s")
            except Exception as e:
                print(f"[TIMING] Could not parse client timestamp: {e}")

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        keyword_extraction_start = datetime.utcnow()
        session = transcription_sessions[session_id]

        if search_mode == "instant":
            keyword = data.get("keyword", "")
        elif search_mode == "recent":
            time_threshold = data.get("time_threshold", 5)
            keyword = session.get_top_keyword_with_time_threshold(time_threshold)
        elif search_mode == "gpt":
            time_threshold = data.get("time_threshold", 10)

            # If Gemini Live is active, wait for next transcription before calling GPT
            if session.gemini_active:
                print(f"[GPT Search] Gemini Live active, setting pending search (words so far: {len(session.words)})")
                session.pending_search = {
                    "timestamp": datetime.utcnow(),
                    "search_type": search_type,
                    "skip_search": skip_search,
                    "show_all_keywords": show_all_keywords,
                    "time_threshold": time_threshold,
                    "words_before": len(session.words),
                    "client_timestamp": client_timestamp,
                    "source": "gemini",
                }
                session._log_event(
                    "pending_search_set",
                    {
                        "words_at_spacebar": len(session.words),
                        "client_timestamp": client_timestamp,
                        "source": "gemini",
                    },
                )
                emit("status", {"message": "Waiting for transcription..."})
                return

            # If Whisper streaming is active, wait for next transcription before calling GPT
            elif session.whisper_active:
                print(f"[GPT Search] Whisper active, setting pending search (words so far: {len(session.words)})")
                session.pending_search = {
                    "timestamp": datetime.utcnow(),
                    "search_type": search_type,
                    "skip_search": skip_search,
                    "show_all_keywords": show_all_keywords,
                    "time_threshold": time_threshold,
                    "words_before": len(session.words),
                    "client_timestamp": client_timestamp,
                    "source": "whisper",
                }
                session._log_event(
                    "pending_search_set",
                    {
                        "words_at_spacebar": len(session.words),
                        "client_timestamp": client_timestamp,
                        "source": "whisper",
                    },
                )
                emit("status", {"message": "Waiting for transcription..."})
                # Start timeout - if no transcription arrives within 2s, process with current text
                _start_pending_search_timeout(session, session_id, socketio, timeout_seconds=2.0)
                return

            keyword = session.get_top_keyword_gpt(time_threshold)
        elif search_mode == "gemini":
            # Use most recently parsed terms (not all accumulated terms)
            if not session.gemini_recent_terms:
                emit(
                    "error",
                    {
                        "message": "No recent Gemini terms available. Wait for terms to be extracted."
                    },
                )
                return

            recent_terms = [
                {"keyword": t["term"], "description": t.get("definition", "")}
                for t in session.gemini_recent_terms
            ]

            if show_all_keywords:
                emit("all_keywords", {"keywords": recent_terms, "mode": "gemini"})
                if skip_search:
                    return

            # Get keyword at current index (for navigation)
            idx = session.gemini_recent_term_index
            if idx >= len(recent_terms):
                idx = 0
                session.gemini_recent_term_index = 0

            keyword = recent_terms[idx]["keyword"]
            description = recent_terms[idx].get("description")

            print(f"[Gemini Search] Using recent term [{idx+1}/{len(recent_terms)}]: {keyword}")
        elif search_mode == "gemini_ondemand":
            # On-demand: query Gemini for keywords when spacebar is pressed
            time_threshold = data.get("time_threshold", 10)
            keyword = session.get_top_keyword_gemini_ondemand(time_threshold)

            if not keyword:
                emit("error", {"message": "Could not extract keywords from Gemini."})
                return

            # Get description from recent_terms if available
            if session.gemini_recent_terms:
                description = session.gemini_recent_terms[0].get("definition", "")

            print(f"[Gemini OnDemand] Keyword: {keyword}")
        else:
            keyword = session.get_top_keyword()

        if not keyword:
            emit("error", {"message": "No keywords available for search"})
            return

        keyword_clean = "".join(
            c for c in keyword if c.isalnum() or c.isspace()
        ).strip()

        description = None
        total_keywords = 1
        current_index = 0

        if search_mode in ("gpt", "gemini") and session.gpt_keyword_pairs:
            total_keywords = len(session.gpt_keyword_pairs)
            current_index = session.current_keyword_index
            if current_index < len(session.gpt_keyword_pairs):
                description = session.gpt_keyword_pairs[current_index].get(
                    "description"
                )

        if show_all_keywords and search_mode == "gpt" and session.gpt_keyword_pairs:
            emit(
                "all_keywords",
                {"keywords": session.gpt_keyword_pairs, "mode": search_mode},
            )
            if skip_search:
                return
        elif search_mode != "gemini" or not show_all_keywords:
            emit(
                "search_keyword",
                {
                    "keyword": keyword_clean,
                    "mode": search_mode,
                    "description": description,
                    "total_keywords": total_keywords,
                    "current_index": current_index,
                },
            )

        if skip_search:
            print("[Search] Skipping Google search (skip_search=True)")
            return

        try:
            search_results = google_custom_search(keyword_clean, search_type)
            num_results = count_results(search_results, search_type)

            session.log_search_action(
                search_mode=search_mode,
                search_type=search_type,
                keyword=keyword_clean,
                num_results=num_results,
            )

            emit(
                "search_results",
                {
                    "keyword": keyword_clean,
                    "mode": search_mode,
                    "type": search_type,
                    "results": search_results,
                },
            )
        except Exception as e:
            print(f"Search error: {str(e)}")
            emit("error", {"message": f"Search error: {str(e)}"})

    @socketio.on("next_keyword")
    def handle_next_keyword(data):
        """Handle double-spacebar request to show next keyword."""
        session_id = request.sid
        search_type = data.get("type", "text")
        search_mode = data.get("mode", "gpt")  # Support both gpt and gemini modes

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]

        # Handle Gemini mode navigation
        if search_mode == "gemini":
            if not session.gemini_recent_terms:
                emit("error", {"message": "No Gemini terms available."})
                return

            session.gemini_recent_term_index += 1
            if session.gemini_recent_term_index >= len(session.gemini_recent_terms):
                session.gemini_recent_term_index = 0

            current_term = session.gemini_recent_terms[session.gemini_recent_term_index]
            keyword = current_term.get("term", "")
            description = current_term.get("definition")

            print(f"[Gemini Next] Term [{session.gemini_recent_term_index+1}/{len(session.gemini_recent_terms)}]: {keyword}")

            emit(
                "search_keyword",
                {
                    "keyword": keyword,
                    "mode": "gemini",
                    "description": description,
                    "total_keywords": len(session.gemini_recent_terms),
                    "current_index": session.gemini_recent_term_index,
                },
            )

            try:
                search_results = google_custom_search(keyword, search_type)
                num_results = count_results(search_results, search_type)

                session.log_search_action(
                    search_mode="gemini_next",
                    search_type=search_type,
                    keyword=keyword,
                    num_results=num_results,
                )

                emit(
                    "search_results",
                    {
                        "keyword": keyword,
                        "mode": "gemini",
                        "type": search_type,
                        "results": search_results,
                    },
                )
            except Exception as e:
                print(f"Search error: {str(e)}")
                emit("error", {"message": f"Search error: {str(e)}"})
            return

        # Handle GPT mode navigation (original logic)
        if not session.gpt_keyword_pairs:
            emit(
                "error", {"message": "No GPT keywords available. Press spacebar first."}
            )
            return

        session.current_keyword_index += 1
        if session.current_keyword_index >= len(session.gpt_keyword_pairs):
            session.current_keyword_index = 0

        current_pair = session.gpt_keyword_pairs[session.current_keyword_index]
        keyword = current_pair.get("keyword", "")
        description = current_pair.get("description")

        emit(
            "search_keyword",
            {
                "keyword": keyword,
                "mode": "gpt",
                "description": description,
                "total_keywords": len(session.gpt_keyword_pairs),
                "current_index": session.current_keyword_index,
            },
        )

        try:
            search_results = google_custom_search(keyword, search_type)
            num_results = count_results(search_results, search_type)

            session.log_search_action(
                search_mode="gpt_next",
                search_type=search_type,
                keyword=keyword,
                num_results=num_results,
            )

            emit(
                "search_results",
                {
                    "keyword": keyword,
                    "mode": "gpt",
                    "type": search_type,
                    "results": search_results,
                },
            )
        except Exception as e:
            print(f"Search error: {str(e)}")
            emit("error", {"message": f"Search error: {str(e)}"})

    @socketio.on("search_single_keyword")
    def handle_search_single_keyword(data):
        """Handle click on a keyword from all_keywords list."""
        session_id = request.sid
        keyword = data.get("keyword", "")
        search_type = data.get("type", "text")

        if not keyword:
            emit("error", {"message": "No keyword provided"})
            return

        emit(
            "search_keyword",
            {
                "keyword": keyword,
                "mode": "gpt",
                "description": None,
                "total_keywords": 1,
                "current_index": 0,
            },
        )

        try:
            search_results = google_custom_search(keyword, search_type)
            num_results = count_results(search_results, search_type)

            if session_id in transcription_sessions:
                transcription_sessions[session_id].log_search_action(
                    search_mode="gpt_single",
                    search_type=search_type,
                    keyword=keyword,
                    num_results=num_results,
                )

            emit(
                "search_results",
                {
                    "keyword": keyword,
                    "mode": "gpt",
                    "type": search_type,
                    "results": search_results,
                },
            )
        except Exception as e:
            print(f"Search error: {str(e)}")
            emit("error", {"message": f"Search error: {str(e)}"})
