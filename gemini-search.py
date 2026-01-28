import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)


grounding_tool = types.Tool(google_search=types.GoogleSearch())

config = types.GenerateContentConfig(tools=[grounding_tool])

import time

before = time.time()

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Who won the euro 2024?",
    config=config,
)

after = time.time()
print(after - before)
# print(response.text)


def add_citations(response):
    text = response.text
    supports = response.candidates[0].grounding_metadata.grounding_supports
    chunks = response.candidates[0].grounding_metadata.grounding_chunks

    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        if support.grounding_chunk_indices:
            # Create citation string like [1](link1)[2](link2)
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    citation_links.append(f"[{i + 1}]({uri})")

            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    return text


# Assuming response with grounding metadata
text_with_citations = add_citations(response)
print(text_with_citations)
