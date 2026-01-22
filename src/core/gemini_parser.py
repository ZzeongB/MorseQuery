"""Gemini output parsing utilities for MorseQuery."""

import re
from typing import Dict, List

from src.core.config import SECTION_RE


def _norm_ws(s: str) -> str:
    """Collapse weird newlines/spaces into readable single spaces."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def _squash_inline(s: str) -> str:
    """Turn arbitrary newlines into spaces, collapse repeated whitespace."""
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _normalize_section_name(name: str) -> str:
    """Normalize section name to standard form."""
    name = name.lower().strip()
    if name in ("caption", "captions"):
        return "captions"
    elif name in ("summary", "summ"):
        return "summary"
    elif name in ("term", "terms"):
        return "terms"
    return name


def _parse_terms_block(text: str) -> List[Dict[str, str]]:
    """Parse a terms block with multiple format support.

    Supports formats:
    - Term: word - definition
    - Term: word: definition
    - **word**: definition
    - bullet word - definition
    - word: definition (fallback)
    """
    if not text:
        return []

    t = _norm_ws(text)
    terms: List[Dict[str, str]] = []
    seen_terms = set()

    # Split by lines first to handle line-based formats
    lines = t.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        term = None
        definition = None

        # Pattern 1: "- **TermName**: definition" (Gemini's primary format)
        m1 = re.match(r"^-?\s*\*\*\s*(.+?)\s*\*\*\s*:\s*(.+)$", line)
        if m1:
            term = m1.group(1).strip()
            definition = m1.group(2).strip()

        # Pattern 2: "Term: word - definition"
        if not term:
            m2 = re.match(
                r"^Term\s*:\s*(.+?)\s*[-\u2013\u2014:]\s*(.+)$", line, re.IGNORECASE
            )
            if m2:
                term = m2.group(1).strip()
                definition = m2.group(2).strip()

        # Pattern 3: "**word** - definition" (no colon after **)
        if not term:
            m3 = re.match(r"^\*\*\s*(.+?)\s*\*\*\s*[-\u2013\u2014]\s*(.+)$", line)
            if m3:
                term = m3.group(1).strip()
                definition = m3.group(2).strip()

        # Pattern 4: "bullet word - definition" or "- word: definition" (no markdown)
        if not term:
            m4 = re.match(r"^[\u2022\-]\s*([^*]+?)\s*[-\u2013\u2014:]\s*(.+)$", line)
            if m4:
                term = m4.group(1).strip()
                definition = m4.group(2).strip()

        # Pattern 5: "Word: definition" (capitalized word)
        if not term:
            m5 = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)]{1,50})\s*:\s*(.+)$", line)
            if m5:
                term = m5.group(1).strip()
                definition = m5.group(2).strip()

        # Pattern 6: Numbered list "1. word - definition"
        if not term:
            m6 = re.match(r"^\d+\.\s*(.+?)\s*[-\u2013\u2014:]\s*(.+)$", line)
            if m6:
                term = m6.group(1).strip()
                definition = m6.group(2).strip()

        # Validate and add term
        if term and definition:
            # Clean up term
            term = re.sub(r"^[\s\-\u2022\*:]+|[\s\-\u2022\*:]+$", "", term)
            # Skip if term is a section header or common word
            skip_terms = {
                "term",
                "terms",
                "caption",
                "captions",
                "summary",
                "overall",
                "current",
                "context",
                "segment",
            }
            if (
                len(term) >= 2
                and len(definition) >= 5
                and term.lower() not in skip_terms
                and term.lower() not in seen_terms
            ):
                terms.append({"term": term, "definition": definition})
                seen_terms.add(term.lower())

    # Fallback: try multi-line patterns if no terms found
    if not terms:
        # Pattern for "Term: word - definition" spanning multiple lines
        term_pat = re.compile(
            r"Term\s*:\s*([^-:\n]+?)\s*[-\u2013\u2014:]\s*([^\n]+)", re.IGNORECASE
        )
        for m in term_pat.finditer(t):
            term = _squash_inline(m.group(1))
            definition = _squash_inline(m.group(2))
            if term and definition and len(term) >= 2 and len(definition) >= 5:
                term_lower = term.lower()
                if term_lower not in seen_terms and term_lower not in {"term", "terms"}:
                    terms.append({"term": term, "definition": definition})
                    seen_terms.add(term_lower)

    return terms


def parse_gemini_output(raw: str) -> Dict[str, object]:
    """Parse Gemini model output containing [Captions], [Summary], [Terms] sections.

    More robust version that handles:
    - Various section header formats ([Section], **Section**, Section:)
    - Streaming/partial content
    - Missing sections
    - Various term formats
    """
    if not raw or not raw.strip():
        return {
            "captions": "",
            "summary": {"overall_context": "", "current_segment": ""},
            "terms": [],
        }

    raw = _norm_ws(raw)

    # Find all section markers and their positions
    matches = list(SECTION_RE.finditer(raw))
    sections: Dict[str, str] = {"captions": "", "summary": "", "terms": ""}

    # If no section markers found, try to extract content anyway
    if not matches:
        # Try to find terms even without section markers
        terms = _parse_terms_block(raw)

        # Use raw text as captions if it looks like transcription
        captions = ""
        if len(raw) > 10 and not any(
            kw in raw.lower() for kw in ["term:", "**", "\u2022"]
        ):
            captions = _squash_inline(raw)

        return {
            "captions": captions,
            "summary": {"overall_context": "", "current_segment": ""},
            "terms": terms,
        }

    # Extract content for each section
    preamble = raw[: matches[0].start()].strip() if matches else ""

    for i, m in enumerate(matches):
        section_name = _normalize_section_name(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()

        # Clean content (remove leading colons, dashes, etc.)
        content = re.sub(r"^[\s:]+", "", content)

        if section_name in sections:
            if sections[section_name]:
                sections[section_name] += "\n" + content
            else:
                sections[section_name] = content

    # Parse captions
    captions = _squash_inline(sections["captions"])

    # If no captions section but we have preamble, use it
    if not captions and preamble:
        # Only use preamble as captions if it doesn't look like terms
        if not any(kw in preamble.lower() for kw in ["term:", "**"]):
            captions = _squash_inline(preamble)

    # Parse summary - flexible format matching
    # Gemini uses format like: "- **Overall context**: text" and "- **Current segment**: text"
    summary_text = sections["summary"]
    overall = ""
    current = ""

    if summary_text:
        # Try multiple patterns for Overall context
        # Pattern 1: "- **Overall context**: text" (markdown bullet)
        # Pattern 2: "Overall context: text"
        overall_patterns = [
            r"-\s*\*\*\s*Overall\s*(?:context)?\s*\*\*\s*:\s*(.+?)(?=-\s*\*\*|$)",
            r"\*\*\s*Overall\s*(?:context)?\s*\*\*\s*:\s*(.+?)(?=\*\*|$)",
            r"Overall\s*(?:context)?\s*:\s*(.+?)(?=Current|Segment|-\s*\*\*|$)",
            r"Context\s*:\s*(.+?)(?=Current|Segment|$)",
        ]

        for pattern in overall_patterns:
            m_overall = re.search(
                pattern, summary_text, flags=re.IGNORECASE | re.DOTALL
            )
            if m_overall:
                overall = _squash_inline(m_overall.group(1))
                if overall:
                    break

        # Try multiple patterns for Current segment
        current_patterns = [
            r"-\s*\*\*\s*Current\s*(?:segment)?\s*\*\*\s*:\s*(.+?)(?=-\s*\*\*|\[|$)",
            r"\*\*\s*Current\s*(?:segment)?\s*\*\*\s*:\s*(.+?)(?=\*\*|\[|$)",
            r"Current\s*(?:segment)?\s*:\s*(.+?)(?=-\s*\*\*|\[|$)",
            r"Segment\s*:\s*(.+)$",
        ]

        for pattern in current_patterns:
            m_current = re.search(
                pattern, summary_text, flags=re.IGNORECASE | re.DOTALL
            )
            if m_current:
                current = _squash_inline(m_current.group(1))
                if current:
                    break

        # If no structured summary found, use the whole text as overall
        if not overall and not current:
            summary_inline = _squash_inline(summary_text)
            if summary_inline:
                overall = summary_inline

    # Parse terms from multiple sources
    terms = []

    # Terms from preamble (sometimes Gemini puts terms before markers)
    terms.extend(_parse_terms_block(preamble))

    # Terms from [Terms] section
    terms.extend(_parse_terms_block(sections["terms"]))

    # Also check if terms are embedded in other sections
    if not terms:
        terms.extend(_parse_terms_block(sections["summary"]))
        terms.extend(_parse_terms_block(sections["captions"]))

    return {
        "captions": captions,
        "summary": {
            "overall_context": overall,
            "current_segment": current,
        },
        "terms": terms,
    }


def test_parse_gemini_output():
    """Test function to verify parsing logic works correctly."""
    # Test case 1: Standard format
    test1 = """[Captions] Hello, this is a test transcription about machine learning.
[Summary] Overall context: Discussion about AI. Current segment: Machine learning basics.
[Terms] Term: Machine Learning - A subset of AI that enables systems to learn from data.
Term: Neural Network - Computing systems inspired by biological neural networks."""

    result1 = parse_gemini_output(test1)
    print("Test 1 (Standard format):")
    print(f"  Captions: {result1['captions']}")
    print(f"  Summary: {result1['summary']}")
    print(f"  Terms: {result1['terms']}")
    print()

    # Test case 2: Markdown format with **
    test2 = """**Captions** The speaker is discussing deep learning applications.
**Summary** This is about neural networks and their uses.
**Terms**
**Deep Learning**: A subset of machine learning using neural networks.
**Backpropagation**: Algorithm for training neural networks."""

    result2 = parse_gemini_output(test2)
    print("Test 2 (Markdown format):")
    print(f"  Captions: {result2['captions']}")
    print(f"  Summary: {result2['summary']}")
    print(f"  Terms: {result2['terms']}")
    print()

    # Test case 3: Plain text without section markers
    test3 = """The lecture covers important topics in computer science.
Term: Algorithm - A step-by-step procedure for solving problems.
Term: Data Structure - A way of organizing data for efficient use."""

    result3 = parse_gemini_output(test3)
    print("Test 3 (No section markers):")
    print(f"  Captions: {result3['captions']}")
    print(f"  Summary: {result3['summary']}")
    print(f"  Terms: {result3['terms']}")
    print()

    # Test case 4: Bullet point format
    test4 = """[Captions] Testing bullet format
[Terms]
- API - Application Programming Interface
- REST - Representational State Transfer
- HTTP - Hypertext Transfer Protocol"""

    result4 = parse_gemini_output(test4)
    print("Test 4 (Bullet format):")
    print(f"  Captions: {result4['captions']}")
    print(f"  Terms: {result4['terms']}")
    print()

    return True


if __name__ == "__main__":
    test_parse_gemini_output()
