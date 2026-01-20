"""
LLM client using OpenRouter API.
"""

import json
import httpx
from schemas import PostBatch, ReplyBatch
from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE


class LLMClient:
    """Client for making LLM API calls via OpenRouter."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/irl-social-agent",  # Required by OpenRouter
        }

    def generate_posts(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = LLM_MODEL,
    ) -> PostBatch:
        """
        Generate social media posts using the LLM.

        Args:
            system_prompt: The system instructions
            user_prompt: The user message with context and topic
            model: The model to use (default from config)

        Returns:
            Validated PostBatch object

        Raises:
            ValueError: If response doesn't match schema
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            # Note: response_format removed - not all models support it
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                self.BASE_URL,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        # Extract the generated content
        content = data["choices"][0]["message"]["content"]

        # Debug: print raw response if empty
        if not content or not content.strip():
            print(f"DEBUG: Empty response from LLM")
            print(f"DEBUG: Full response: {data}")
            raise ValueError("LLM returned empty response")

        # Parse and validate with Pydantic
        return self._parse_response(content)

    def generate_replies(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = LLM_MODEL,
    ) -> ReplyBatch:
        """
        Generate replies to posts using the LLM.

        Args:
            system_prompt: The system instructions
            user_prompt: The user message with posts to reply to
            model: The model to use (default from config)

        Returns:
            Validated ReplyBatch object
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            # Note: response_format removed - not all models support it
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                self.BASE_URL,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        return self._parse_reply_response(content)

    def _extract_json(self, content: str) -> str:
        """Extract JSON from response, handling markdown code blocks."""
        content = content.strip()

        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()

        return content

    def _parse_response(self, content: str) -> PostBatch:
        """
        Parse and validate LLM response into PostBatch.

        Args:
            content: Raw JSON string from LLM

        Returns:
            Validated PostBatch object

        Raises:
            ValueError: If parsing or validation fails
        """
        try:
            # Extract JSON (handles markdown code blocks)
            json_str = self._extract_json(content)

            # Parse JSON
            raw_data = json.loads(json_str)

            # Validate with Pydantic
            return PostBatch.model_validate(raw_data)

        except json.JSONDecodeError as e:
            print(f"DEBUG: Failed to parse JSON. Raw content:\n{content[:500]}")
            raise ValueError(f"LLM returned invalid JSON: {e}")
        except Exception as e:
            raise ValueError(f"Response validation failed: {e}")

    def _parse_reply_response(self, content: str) -> ReplyBatch:
        """
        Parse and validate LLM response into ReplyBatch.

        Args:
            content: Raw JSON string from LLM

        Returns:
            Validated ReplyBatch object
        """
        try:
            json_str = self._extract_json(content)
            raw_data = json.loads(json_str)
            return ReplyBatch.model_validate(raw_data)
        except json.JSONDecodeError as e:
            print(f"DEBUG: Failed to parse JSON. Raw content:\n{content[:500]}")
            raise ValueError(f"LLM returned invalid JSON: {e}")
        except Exception as e:
            raise ValueError(f"Response validation failed: {e}")
