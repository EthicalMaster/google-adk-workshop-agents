import os

from ddgs import DDGS

from google.adk.agents.llm_agent import Agent
from google.adk.integrations.openai import OpenAILlm


def web_search(search_query: str) -> dict:
    """
    Search the web for current or external information.

    Args:
        search_query: The user's search query.

    Returns:
        A dictionary containing relevant web search results.
    """
    try:
        results = DDGS().text(
            search_query,
            max_results=5,
        )

        if not results:
            return {
                "status": "success",
                "query": search_query,
                "results": [],
                "message": "No search results were found."
            }

        formatted_results = []

        for result in results:
            formatted_results.append(
                {
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "url": result.get("href", ""),
                }
            )

        return {
            "status": "success",
            "query": search_query,
            "results": formatted_results,
        }

    except Exception as error:
        return {
            "status": "error",
            "query": search_query,
            "message": str(error),
        }


def calculator(expression: str) -> dict:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression as a string.
                    Example: "12500 * 18 / 100"

    Returns:
        A dictionary containing the result or an error message.
    """
    try:
        allowed_characters = set("0123456789+-*/().% ")

        if not set(expression).issubset(allowed_characters):
            return {
                "status": "error",
                "message": "The expression contains unsupported characters."
            }

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return {
            "status": "success",
            "expression": expression,
            "result": result
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }


def text_analyzer(text: str) -> dict:
    """
    Analyze text and return character count, word count,
    and sentence count.

    Args:
        text: Text to analyze.

    Returns:
        A dictionary containing text statistics.
    """
    character_count = len(text)
    word_count = len(text.split())

    sentence_endings = {".", "!", "?"}

    sentence_count = sum(
        1
        for character in text
        if character in sentence_endings
    )

    return {
        "status": "success",
        "character_count": character_count,
        "word_count": word_count,
        "sentence_count": sentence_count
    }


groq_model = OpenAILlm(
    model="qwen/qwen3.8-27b",
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


root_agent = Agent(
    name="personal_ai_assistant",

    model=groq_model,

    description=(
        "A general-purpose personal AI assistant that can search the web, "
        "perform mathematical calculations, and analyze text."
    ),

    instruction="""
You are a helpful general-purpose Personal AI Assistant.

Understand the user's request and automatically choose the correct
tool or tools.

AVAILABLE TOOLS

1. web_search

Use web_search whenever the user asks for:
- current information
- latest information
- recent developments
- external information
- information from the web
- information that may have changed recently

Use the search results returned by the tool to answer the user.
Do not claim that you searched the web unless the web_search tool
was actually used.

2. calculator

Use calculator whenever the user asks for:
- mathematical calculations
- arithmetic
- percentages
- GST calculations
- numerical expressions

Do not perform calculations manually when the calculator tool is
appropriate.

3. text_analyzer

Use text_analyzer whenever the user asks for:
- character count
- word count
- sentence count
- text statistics

Do not estimate these values manually.

MULTI-TOOL REQUESTS

A single user request may require more than one tool.

For example:

"Search for information about Google ADK and calculate 18% of ₹50,000."

For this request:
- use web_search to find information about Google ADK
- use calculator to calculate 18% of ₹50,000
- combine both results into one final response

Use every tool necessary to completely answer the user.

After all required tools finish, provide one clear final response.

Do not expose internal reasoning.
""",

    tools=[
        web_search,
        calculator,
        text_analyzer,
    ],
)