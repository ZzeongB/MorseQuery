"""Benchmark script to compare Gemini and GPT model response times."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from google import genai
from google.genai import types

# Initialize clients
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Models to test
GEMINI_MODELS = [
    # "gemini-2.0-flash",
    # "gemini-2.0-flash-lite",
    # "gemini-3-flash-preview",
]

# GPT-4 series (chat.completions API)
GPT4_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
]

# GPT-5 series (responses API)
GPT5_MODELS = [
    "gpt-5-mini",
    "gpt-5-nano",
]

# Test prompt (same as keyword extraction)
TEST_PROMPT = """You are analyzing transcripts. Users listen to content and select specific words or phrases they want to look up.

Given the transcript context, predict the top three words or phrases the user would most likely want to look up. The selected words or phrases should be:
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
Description: <a brief 1-sentence description>

Context: The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow models to weigh the importance of different parts of the input sequence."""

NUM_RUNS = 3  # Number of runs per model


def benchmark_gemini(model_name: str) -> dict:
    """Benchmark a Gemini model."""
    times = []
    errors = []

    for i in range(NUM_RUNS):
        try:
            start = time.perf_counter()

            response = gemini_client.models.generate_content(
                model=model_name,
                contents=TEST_PROMPT,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                ),
            )
            _ = response.text  # Ensure response is received

            end = time.perf_counter()
            elapsed = end - start
            times.append(elapsed)

            print(f"  Run {i+1}: {elapsed:.3f}s")

        except Exception as e:
            errors.append(str(e))
            print(f"  Run {i+1}: ERROR - {e}")

    return {
        "model": model_name,
        "provider": "Gemini",
        "times": times,
        "avg": sum(times) / len(times) if times else None,
        "min": min(times) if times else None,
        "max": max(times) if times else None,
        "errors": errors,
    }


def benchmark_gpt4(model_name: str) -> dict:
    """Benchmark a GPT-4 model using chat.completions API."""
    if not openai_client:
        return {
            "model": model_name,
            "provider": "OpenAI",
            "times": [],
            "avg": None,
            "min": None,
            "max": None,
            "errors": ["OpenAI API key not configured"],
        }

    times = []
    errors = []

    for i in range(NUM_RUNS):
        try:
            start = time.perf_counter()

            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                temperature=0.3,
            )
            _ = response.choices[0].message.content  # Ensure response is received

            end = time.perf_counter()
            elapsed = end - start
            times.append(elapsed)

            print(f"  Run {i+1}: {elapsed:.3f}s")

        except Exception as e:
            errors.append(str(e))
            print(f"  Run {i+1}: ERROR - {e}")

    return {
        "model": model_name,
        "provider": "OpenAI",
        "times": times,
        "avg": sum(times) / len(times) if times else None,
        "min": min(times) if times else None,
        "max": max(times) if times else None,
        "errors": errors,
    }


def benchmark_gpt5(model_name: str) -> dict:
    """Benchmark a GPT-5 model using responses API."""
    if not openai_client:
        return {
            "model": model_name,
            "provider": "OpenAI",
            "times": [],
            "avg": None,
            "min": None,
            "max": None,
            "errors": ["OpenAI API key not configured"],
        }

    times = []
    errors = []

    for i in range(NUM_RUNS):
        try:
            start = time.perf_counter()

            response = openai_client.responses.create(
                model=model_name,
                input=TEST_PROMPT,
            )
            _ = response.output_text  # Ensure response is received

            end = time.perf_counter()
            elapsed = end - start
            times.append(elapsed)

            print(f"  Run {i+1}: {elapsed:.3f}s")

        except Exception as e:
            errors.append(str(e))
            print(f"  Run {i+1}: ERROR - {e}")

    return {
        "model": model_name,
        "provider": "OpenAI",
        "times": times,
        "avg": sum(times) / len(times) if times else None,
        "min": min(times) if times else None,
        "max": max(times) if times else None,
        "errors": errors,
    }


def main():
    print("=" * 70)
    print("LLM Model Benchmark (Gemini vs GPT)")
    print("=" * 70)
    print(f"Runs per model: {NUM_RUNS}")
    print(f"Prompt length: {len(TEST_PROMPT)} chars")
    print("=" * 70)

    results = []

    # Benchmark Gemini models
    print("\n[GEMINI MODELS]")
    for model in GEMINI_MODELS:
        print(f"\nTesting: {model}")
        print("-" * 40)
        result = benchmark_gemini(model)
        results.append(result)

    # Benchmark GPT-4 models (chat.completions API)
    print("\n[GPT-4 MODELS]")
    for model in GPT4_MODELS:
        print(f"\nTesting: {model}")
        print("-" * 40)
        result = benchmark_gpt4(model)
        results.append(result)

    # Benchmark GPT-5 models (responses API)
    print("\n[GPT-5 MODELS]")
    for model in GPT5_MODELS:
        print(f"\nTesting: {model}")
        print("-" * 40)
        result = benchmark_gpt5(model)
        results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Provider':<10} {'Avg':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)

    # Sort by average time
    valid_results = [r for r in results if r["avg"] is not None]
    valid_results.sort(key=lambda x: x["avg"])

    for r in valid_results:
        print(
            f"{r['model']:<25} {r['provider']:<10} {r['avg']:>7.3f}s {r['min']:>7.3f}s {r['max']:>7.3f}s"
        )

    # Print failed models
    failed = [r for r in results if r["avg"] is None]
    for r in failed:
        error_msg = r["errors"][0][:20] if r["errors"] else "Unknown"
        print(f"{r['model']:<25} {r['provider']:<10} FAILED ({error_msg}...)")

    print("=" * 70)

    if valid_results:
        fastest = valid_results[0]
        print(
            f"\nFastest: {fastest['model']} ({fastest['provider']}) - {fastest['avg']:.3f}s avg"
        )


if __name__ == "__main__":
    main()
