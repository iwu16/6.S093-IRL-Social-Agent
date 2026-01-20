"""
Structured output schemas for social media posts.
Uses Pydantic for validation and JSON schema generation.
"""

from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


class PostTone(str, Enum):
    """Allowed tone values for IRL brand voice."""
    CALM = "calm"
    WARM = "warm"
    REFLECTIVE = "reflective"
    INVITING = "inviting"
    UNDERSTATED = "understated"


class IntendedUse(str, Enum):
    """Categories for how the post should be used."""
    BRAND_VOICE = "brand_voice"
    ENGAGEMENT = "engagement"
    ANNOUNCEMENT = "announcement"
    CONVERSATION = "conversation"
    AWARENESS = "awareness"


class SocialPost(BaseModel):
    """A single social media post with metadata."""

    text: str = Field(
        ...,
        description="The post content (max 500 chars for Mastodon)",
        max_length=500
    )
    tone: PostTone = Field(
        ...,
        description="The emotional tone of the post"
    )
    intended_use: IntendedUse = Field(
        ...,
        description="How this post should be used in the content strategy"
    )


class PostBatch(BaseModel):
    """A batch of generated posts for review."""

    topic: str = Field(
        ...,
        description="The topic/theme these posts were generated for"
    )
    posts: list[SocialPost] = Field(
        ...,
        description="List of 3-5 generated posts",
        min_length=3,
        max_length=5
    )


class Reply(BaseModel):
    """A reply to a specific post."""

    original_post_id: str = Field(
        ...,
        description="The ID of the post being replied to"
    )
    reply_text: str = Field(
        ...,
        description="The reply content (max 500 chars for Mastodon)",
        max_length=500
    )
    tone: PostTone = Field(
        ...,
        description="The emotional tone of the reply"
    )


class ReplyBatch(BaseModel):
    """A batch of generated replies to searched posts."""

    keyword: str = Field(
        ...,
        description="The keyword used to find these posts"
    )
    replies: list[Reply] = Field(
        ...,
        description="List of replies to the found posts",
        min_length=1,
        max_length=5
    )


# JSON Schema for reference (can be used with OpenAI-compatible APIs)
POST_BATCH_SCHEMA = PostBatch.model_json_schema()
REPLY_BATCH_SCHEMA = ReplyBatch.model_json_schema()


# Example conforming response for testing/demo
SAMPLE_RESPONSE = {
    "topic": "spontaneous plans",
    "posts": [
        {
            "text": "Sometimes the best plans are the ones you didn't make. A walk with someone new. A coffee that turns into a conversation. That's what we're here for.",
            "tone": "calm",
            "intended_use": "brand_voice"
        },
        {
            "text": "You don't need a reason to go outside. You just need a direction.",
            "tone": "reflective",
            "intended_use": "engagement"
        },
        {
            "text": "Real connection doesn't require a calendar invite.",
            "tone": "understated",
            "intended_use": "brand_voice"
        },
        {
            "text": "What if today you said yes to something small? A study session. A neighborhood walk. A moment that wasn't planned.",
            "tone": "inviting",
            "intended_use": "conversation"
        },
        {
            "text": "We built IRL because we missed the spontaneous. The let's-just-go. The why-not.",
            "tone": "warm",
            "intended_use": "awareness"
        }
    ]
}
