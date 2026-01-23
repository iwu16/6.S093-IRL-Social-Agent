"""
Mastodon Mention Listener for IRL Social Agent.

Polls Mastodon for mentions/replies and automatically:
1. Generates contextual replies using RAG
2. Sends replies to Telegram for HITL approval
3. Posts approved replies to Mastodon
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import Config
from database import Database
from mastodon_client import MastodonClient, MastodonPost
from rag_client import RAGClient
from llm_client import LLMClient
from prompts import SYSTEM_PROMPT


# Cache file for tracking processed mentions
CACHE_FILE = Path(__file__).parent / ".mastodon_mentions_cache.json"

# Prompt template for generating replies
REPLY_SYSTEM_PROMPT = """You are a social media manager for IRL (Sidequest), a mobile app that helps people join small, real-world activities happening right now in their city.

## Brand Voice Guidelines

IRL's voice is:
- Calm and understated (never hype or urgent)
- Human and warm (like a thoughtful friend, not a brand)
- Reflective (invites people to think, not react)
- Grounded in real life (not digital-first language)

## Writing Rules for Replies

1. NO emojis ever
2. NO marketing buzzwords
3. Be genuine and conversational
4. Address what the person actually said
5. Keep replies short (1-2 sentences max)
6. Don't be pushy or promotional
7. It's okay to just be friendly without mentioning the app
"""

REPLY_PROMPT_TEMPLATE = """## Context About Our Company

{context}

---

## The Mention We Received

From: {author} ({handle})
Their message: {content}
URL: {url}

---

## Task

Write a thoughtful, genuine reply to this mention. The reply should:
- Actually respond to what they said (not generic)
- Feel like a real person wrote it
- Match IRL's calm, human brand voice
- Be 1-2 sentences max
- Include their @handle at the start

Return ONLY the reply text, nothing else. Start with @{handle_clean}
"""


class MastodonListener:
    """Listens for Mastodon mentions and generates replies with HITL approval."""

    def __init__(
        self,
        config: Config,
        poll_interval: int = 60,  # 1 minute
        auto_approve: bool = False,
    ):
        """
        Initialize the Mastodon listener.

        Args:
            config: Application configuration
            poll_interval: Seconds between polls (default: 60 = 1 min)
            auto_approve: If True, post replies without HITL (use carefully!)
        """
        self.config = config
        self.poll_interval = poll_interval
        self.auto_approve = auto_approve
        self.mastodon = MastodonClient(
            token=config.mastodon_token,
            instance=config.mastodon_instance,
        )
        self.rag = RAGClient()
        self.llm = LLMClient(api_key=config.openrouter_api_key)
        self.db = Database()
        self._processed_mentions: set[str] = set()
        self._load_cache()

    def _load_cache(self):
        """Load processed mention IDs from cache."""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE) as f:
                    data = json.load(f)
                    self._processed_mentions = set(data.get("processed", []))
                print(f"Loaded {len(self._processed_mentions)} processed mention IDs")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")

    def _save_cache(self):
        """Save processed mention IDs to cache."""
        try:
            # Keep only last 1000 mentions to prevent unbounded growth
            recent = list(self._processed_mentions)[-1000:]
            with open(CACHE_FILE, "w") as f:
                json.dump({"processed": recent}, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _mark_processed(self, mention_id: str):
        """Mark a mention as processed."""
        self._processed_mentions.add(mention_id)
        self._save_cache()

    def fetch_mentions(self) -> list[MastodonPost]:
        """
        Fetch new mentions from Mastodon.

        Returns:
            List of unprocessed mentions
        """
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking for new mentions...")

        try:
            mentions = self.mastodon.get_mentions(limit=20)
        except Exception as e:
            print(f"Error fetching mentions: {e}")
            return []

        # Filter out already processed mentions
        new_mentions = [
            m for m in mentions
            if m.id not in self._processed_mentions
        ]

        if new_mentions:
            print(f"  Found {len(new_mentions)} new mentions")
        else:
            print("  No new mentions")

        return new_mentions

    def generate_reply(self, mention: MastodonPost) -> Optional[str]:
        """
        Generate a reply to a mention using RAG context.

        Args:
            mention: The mention to reply to

        Returns:
            Generated reply text, or None if generation failed
        """
        print(f"  Generating reply to @{mention.author_handle}...")

        # Get relevant context from RAG
        # Use the mention content as the query
        context = self.rag.search(mention.content, top_k=3)

        if not context or context == "No relevant context found.":
            context = "IRL (Sidequest) helps people join small, real-world activities in their city."

        # Build the prompt
        handle_clean = mention.author_handle.lstrip("@")
        prompt = REPLY_PROMPT_TEMPLATE.format(
            context=context,
            author=mention.author,
            handle=mention.author_handle,
            handle_clean=handle_clean,
            content=mention.content,
            url=mention.url,
        )

        try:
            # Generate reply using LLM
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": REPLY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.7,
            )

            reply_text = response.choices[0].message.content.strip()

            # Ensure it starts with the handle
            if not reply_text.startswith("@"):
                reply_text = f"@{handle_clean} {reply_text}"

            print(f"    Generated: {reply_text[:80]}...")
            return reply_text

        except Exception as e:
            print(f"    Error generating reply: {e}")
            return None

    async def request_approval(self, mention: MastodonPost, reply_text: str) -> tuple[bool, str]:
        """
        Request HITL approval via Telegram.

        Args:
            mention: The original mention
            reply_text: The generated reply

        Returns:
            Tuple of (approved, final_text) - final_text may be edited
        """
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            print("    Telegram not configured, auto-approving")
            return True, reply_text

        from telegram_hitl import TelegramHITL

        hitl = TelegramHITL(
            bot_token=self.config.telegram_bot_token,
            chat_id=self.config.telegram_chat_id,
        )

        # Format the approval request
        message = (
            f"New mention from @{mention.author_handle}:\n\n"
            f'"{mention.content[:200]}..."\n\n'
            f"Proposed reply:\n"
            f'"{reply_text}"'
        )

        result = await hitl.request_approval(
            post_text=reply_text,
            post_index=1,
            total_posts=1,
            topic=f"Reply to @{mention.author_handle}",
            tone="warm",
        )

        if result.decision == "approve":
            return True, reply_text
        elif result.decision == "edit":
            return True, result.edited_text
        else:
            print(f"    Rejected: {result.rejection_reason}")
            return False, ""

    def post_reply(self, mention: MastodonPost, reply_text: str) -> bool:
        """
        Post a reply to Mastodon.

        Args:
            mention: The mention being replied to
            reply_text: The reply text

        Returns:
            True if posted successfully
        """
        try:
            result = self.mastodon.reply_to_post(
                text=reply_text,
                reply_to_id=mention.id,
            )

            if result.success:
                print(f"    Posted reply: {result.url}")
                return True
            else:
                print(f"    Failed to post: {result.error}")
                return False

        except Exception as e:
            print(f"    Error posting reply: {e}")
            return False

    async def process_mention(self, mention: MastodonPost) -> bool:
        """
        Process a single mention: generate reply, get approval, post.

        Returns:
            True if reply was posted successfully
        """
        print(f"\nProcessing mention from @{mention.author_handle}")
        print(f"  Content: {mention.content[:100]}...")

        # Generate reply
        reply_text = self.generate_reply(mention)
        if not reply_text:
            self._mark_processed(mention.id)
            return False

        # Get approval (unless auto_approve is enabled)
        if self.auto_approve:
            approved, final_text = True, reply_text
        else:
            approved, final_text = await self.request_approval(mention, reply_text)

        if not approved:
            self._mark_processed(mention.id)
            return False

        # Post the reply
        success = self.post_reply(mention, final_text)
        self._mark_processed(mention.id)

        return success

    async def run_once(self) -> int:
        """
        Run a single check for mentions and process them.

        Returns:
            Number of replies posted
        """
        mentions = self.fetch_mentions()
        replies_posted = 0

        for mention in mentions:
            if await self.process_mention(mention):
                replies_posted += 1

        return replies_posted

    async def run(self):
        """Run the listener continuously."""
        print(f"Starting Mastodon listener (polling every {self.poll_interval}s)")
        print(f"Auto-approve replies: {self.auto_approve}")
        print("Press Ctrl+C to stop\n")

        # Verify credentials first
        if not self.mastodon.verify_credentials():
            print("Failed to verify Mastodon credentials. Check your token.")
            return

        print(f"Connected to {self.config.mastodon_instance}")

        try:
            while True:
                await self.run_once()
                await asyncio.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\nStopping listener...")


# Add get_mentions method to MastodonClient
def add_get_mentions(mastodon_client_class):
    """Add method to MastodonClient for fetching mentions."""
    def get_mentions(self, limit: int = 20) -> list:
        """
        Fetch mentions (notifications where someone mentioned this account).

        Returns:
            List of MastodonPost objects representing mentions
        """
        import re
        import httpx

        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.api_base}/notifications",
                    params={"types[]": "mention", "limit": limit},
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=30.0,
                )
                response.raise_for_status()

            mentions = []
            for notif in response.json():
                if notif.get("type") == "mention" and notif.get("status"):
                    status = notif["status"]
                    account = status.get("account", {})

                    # Strip HTML tags from content
                    content = re.sub(r'<[^>]+>', '', status.get("content", ""))

                    mentions.append(MastodonPost(
                        id=status["id"],
                        content=content,
                        author=account.get("display_name", account.get("username", "Unknown")),
                        author_handle=f"@{account.get('acct', 'unknown')}",
                        url=status.get("url", ""),
                        created_at=status.get("created_at", ""),
                    ))

            return mentions

        except Exception as e:
            print(f"Error fetching mentions: {e}")
            return []

    mastodon_client_class.get_mentions = get_mentions


# Monkey-patch MastodonClient with the new method
add_get_mentions(MastodonClient)


async def main():
    parser = argparse.ArgumentParser(description="Listen for Mastodon mentions and auto-reply")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds (default: 60 = 1 min)"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically post replies without HITL approval (use carefully!)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't poll continuously)"
    )

    args = parser.parse_args()

    config = Config.from_env()

    if not config.mastodon_token:
        print("Error: MASTODON_TOKEN not configured")
        return

    listener = MastodonListener(
        config=config,
        poll_interval=args.interval,
        auto_approve=args.auto_approve,
    )

    if args.once:
        replies = await listener.run_once()
        print(f"\nReplies posted: {replies}")
    else:
        await listener.run()


if __name__ == "__main__":
    asyncio.run(main())
