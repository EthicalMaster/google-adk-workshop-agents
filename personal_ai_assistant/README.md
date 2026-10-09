# Personal AI Assistant - Task 1

This project was created for the NPTEL workshop assignment
"Beyond Chatbots: Building AI Agents with Google ADK".

## Features

The assistant can automatically select between three tools:

1. Web Search
   - Searches for current or external information.
   - Implemented using DDGS.

2. Calculator
   - Performs mathematical calculations and percentage calculations.

3. Text Analyzer
   - Returns character count, word count and sentence count.

The agent can also use more than one tool for the same request.

## Model

The agent uses a Groq-hosted model through Google ADK's
OpenAI-compatible model integration.

Model:

qwen/qwen3.8-27b

## Run

Install the required packages:

```bash
pip install -r requirements.txt