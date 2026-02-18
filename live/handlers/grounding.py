"""Search grounding handler using Gemini API."""

from typing import Any

from config import GOOGLE_API_KEY
from flask_socketio import SocketIO
from logger import get_logger, log_print

# Prompts for different detail levels
DETAIL_LEVEL_PROMPTS = {
    1: (
        "Provide a detailed explanation in  2-3 sentences about: "
        "{keyword}{description_suffix}. "
        "Focus on the most essential definition or meaning. "
        "Use **bold** sparingly for key terms only."
    ),
    2: (
        "Provide a detailed explanation in 2-3 sentences about: "
        "{keyword}{description_suffix}. "
        "Include key characteristics, context, and significance. "
        "Use **bold** for important terms and concepts."
    ),
    3: (
        "Provide comprehensive information in 4 or more sentences about: "
        "{keyword}{description_suffix}. "
        "Include: detailed explanation, historical context or background, "
        "practical examples or applications, and related concepts. "
        "Use **bold** for important terms. Be thorough but clear."
    ),
}


def handle_search_grounding(
    sio: SocketIO, session_id: str, data: dict[str, Any]
) -> None:
    """Handle long-press search grounding request using Gemini API."""
    from google import genai
    from google.genai import types

    keyword = data.get("keyword", "")
    description = data.get("description", "")
    detail_level = data.get("detail_level", 1)

    if not keyword:
        sio.emit("error", {"message": "No keyword provided for grounding search"})
        return

    log_print(
        "INFO",
        "Search grounding requested",
        session_id=session_id,
        keyword=keyword,
        description=description,
        detail_level=detail_level,
    )
    logger = get_logger(session_id)
    logger.log(
        "search_grounding_request",
        keyword=keyword,
        description=description,
        detail_level=detail_level,
    )

    if not GOOGLE_API_KEY:
        sio.emit("error", {"message": "GOOGLE_API_KEY not configured"})
        return

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        description_suffix = f" ({description})" if description else ""
        prompt_template = DETAIL_LEVEL_PROMPTS.get(
            detail_level, DETAIL_LEVEL_PROMPTS[1]
        )
        prompt = prompt_template.format(
            keyword=keyword, description_suffix=description_suffix
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=config,
        )

        text = response.text
        citations = {
            "text": text,
            "citations": [],
        }  # _extract_citations(response, text)

        log_print(
            "INFO",
            f"Grounding response: {text[:100]}...",
            session_id=session_id,
            citations=len(citations["citations"]),
        )
        logger.log(
            "search_grounding_result",
            keyword=keyword,
            text_length=len(citations["text"]),
            citations_count=len(citations["citations"]),
        )

        sio.emit(
            "grounding_result",
            {
                "keyword": keyword,
                "text": citations["text"],
                "citations": citations["citations"],
                "detail_level": detail_level,
            },
        )

    except Exception as e:
        log_print("ERROR", f"Search grounding error: {str(e)}", session_id=session_id)
        logger.log("search_grounding_error", error=str(e))
        sio.emit("error", {"message": f"Search grounding error: {str(e)}"})


def _extract_citations(response: Any, text: str) -> dict[str, Any]:
    """Extract citations from Gemini response and insert into text."""
    citations: list[dict] = []

    if not (
        response.candidates
        and response.candidates[0].grounding_metadata
        and response.candidates[0].grounding_metadata.grounding_supports
    ):
        return {"text": text, "citations": citations}

    supports = response.candidates[0].grounding_metadata.grounding_supports
    chunks = response.candidates[0].grounding_metadata.grounding_chunks

    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        if not support.grounding_chunk_indices:
            continue

        citation_refs = []
        for i in support.grounding_chunk_indices:
            if i >= len(chunks):
                continue

            uri = chunks[i].web.uri
            title = getattr(chunks[i].web, "title", f"Source {i + 1}")
            existing = next((c for c in citations if c["uri"] == uri), None)

            if not existing:
                citations.append(
                    {"index": len(citations) + 1, "uri": uri, "title": title}
                )
                citation_refs.append(len(citations))
            else:
                citation_refs.append(existing["index"])

        citation_string = "".join([f"[{ref}]" for ref in citation_refs])
        text = text[:end_index] + citation_string + text[end_index:]

    return {"text": text, "citations": citations}
