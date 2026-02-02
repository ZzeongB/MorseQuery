"""Prompt configuration presets for MorseQuery."""

# Configuration presets for OpenAI transcription and Gemini search prompts
# Each configuration can have custom prompts/keywords to guide the AI

PROMPT_CONFIGS = {
    1: {
        "name": "Default",
        "description": "Standard configuration",
        "openai_transcription_prompt": "",
        "openai_transcription_language": "en",
        "gemini_search_prompt_prefix": "",
    },
    2: {
        "name": "Technical",
        "description": "For technical content (programming, engineering)",
        "openai_transcription_prompt": "Technical terms, programming languages, frameworks, APIs",
        "openai_transcription_language": "en",
        "gemini_search_prompt_prefix": "Focus on technical details and implementation. ",
    },
    3: {
        "name": "Medical",
        "description": "For medical/health content",
        "openai_transcription_prompt": "Medical terminology, drug names, conditions, treatments",
        "openai_transcription_language": "en",
        "gemini_search_prompt_prefix": "Focus on medical accuracy and clinical information. ",
    },
    4: {
        "name": "Business",
        "description": "For business/finance content",
        "openai_transcription_prompt": "Business terms, company names, financial terminology",
        "openai_transcription_language": "en",
        "gemini_search_prompt_prefix": "Focus on business context and market information. ",
    },
    5: {
        "name": "Academic",
        "description": "For academic/research content",
        "openai_transcription_prompt": "Academic terms, research methodology, citations, theories",
        "openai_transcription_language": "en",
        "gemini_search_prompt_prefix": "Focus on academic sources and scholarly context. ",
    },
    6: {
        "name": "Korean",
        "description": "For Korean language content",
        "openai_transcription_prompt": "",
        "openai_transcription_language": "ko",
        "gemini_search_prompt_prefix": "Respond in Korean when appropriate. ",
    },
}


def get_config(config_id: int) -> dict:
    """Get configuration by ID. Returns default if not found."""
    return PROMPT_CONFIGS.get(config_id, PROMPT_CONFIGS[1])


def get_all_configs() -> dict:
    """Get all configuration presets."""
    return PROMPT_CONFIGS
