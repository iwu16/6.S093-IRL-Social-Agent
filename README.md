# IRL Social Media Agent

A prototype AI agent for MIT 6.S093 that generates social media posts using company documents from Notion as context.

## Overview

This agent:
1. Reads company documents from Notion via API
2. Uses them as RAG-style context for an LLM
3. Generates structured social media posts
4. Outputs posts for **manual review** (no auto-posting)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

## Usage

```bash
python main.py --topic "spontaneous plans"
```

## Project Structure

```
irl-social-agent/
├── main.py              # Entry point
├── config.py            # Environment and settings
├── notion_client.py     # Notion API integration
├── llm_client.py        # LLM API calls (OpenRouter)
├── schemas.py           # Pydantic models for structured output
├── prompts.py           # System prompts and templates
├── requirements.txt
└── .env.example
```
