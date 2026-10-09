# SmartStudy Multi-Agent Planner - Task 2

This project was created for the NPTEL workshop assignment
"Beyond Chatbots: Building AI Agents with Google ADK".

SmartStudy creates a personalized study plan using a multi-agent
workflow.

## Workflow

### Sequential workflow

The main workflow follows:

Coordinator Agent
→ Planning Agent
→ Parallel Analysis
→ Study Plan Writer
→ Review Loop
→ Final Plan

### Parallel workflow

Three agents independently analyze the student's requirements:

- Topic Agent
- Practice Agent
- Schedule Agent

Their outputs are stored separately and combined by the
Study Plan Writer.

### Loop workflow

The generated study plan is reviewed and revised for a maximum
of two iterations.

The reviewer checks:

- topic coverage
- available study time
- practice
- revision
- final-day preparation

## Model

Gemini 3.5 Flash Lite

## Run

Install dependencies:

```bash
pip install -r requirements.txt