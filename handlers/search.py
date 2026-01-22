"""Search-related SocketIO event handlers."""

from datetime import datetime

from flask import request
from flask_socketio import emit

from src.core.search import count_results, google_custom_search


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
            time_threshold = data.get("time_threshold", 5)
            keyword = session.get_top_keyword_gpt(time_threshold)
        elif search_mode == "gemini":
            if not session.gemini_terms:
                emit(
                    "error",
                    {
                        "message": "No Gemini terms available. Start Gemini Live first and wait for terms to be extracted."
                    },
                )
                return

            gemini_terms = session.get_gemini_terms_for_search()

            if show_all_keywords:
                emit("all_keywords", {"keywords": gemini_terms, "mode": "gemini"})
                if skip_search:
                    return
                keyword = gemini_terms[0]["keyword"] if gemini_terms else ""
            else:
                session.gpt_keyword_pairs = gemini_terms
                session.current_keyword_index = 0
                keyword = gemini_terms[0]["keyword"] if gemini_terms else ""
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
        """Handle double-spacebar request to show next GPT keyword."""
        session_id = request.sid
        search_type = data.get("type", "text")

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]

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
