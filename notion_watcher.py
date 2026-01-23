"""
Notion Document Watcher for IRL Social Agent.

Polls Notion for document changes and automatically:
1. Re-embeds modified documents into RAG
2. Optionally generates posts about new/updated content
"""

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import Config
from database import Database
from notion_client import NotionClient, NotionPage
from rag_client import RAGClient


# Cache file for tracking document hashes
CACHE_FILE = Path(__file__).parent / ".notion_cache.json"


class NotionWatcher:
    """Watches Notion for document changes and triggers re-embedding."""

    def __init__(
        self,
        config: Config,
        poll_interval: int = 300,  # 5 minutes
        auto_generate: bool = False,
    ):
        """
        Initialize the Notion watcher.

        Args:
            config: Application configuration
            poll_interval: Seconds between polls (default: 300 = 5 min)
            auto_generate: Whether to auto-generate posts for new/changed docs
        """
        self.config = config
        self.poll_interval = poll_interval
        self.auto_generate = auto_generate
        self.notion = NotionClient(token=config.notion_token)
        self.rag = RAGClient()
        self.db = Database()
        self._document_hashes: dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        """Load document hashes from cache file."""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE) as f:
                    self._document_hashes = json.load(f)
                print(f"Loaded {len(self._document_hashes)} cached document hashes")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self._document_hashes = {}

    def _save_cache(self):
        """Save document hashes to cache file."""
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(self._document_hashes, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _compute_hash(self, doc: NotionPage) -> str:
        """Compute hash of document content for change detection."""
        content = f"{doc.title}:{doc.content}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_for_changes(self) -> tuple[list[NotionPage], list[NotionPage], list[str]]:
        """
        Check Notion for document changes.

        Returns:
            Tuple of (new_docs, modified_docs, deleted_doc_ids)
        """
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking Notion for changes...")

        # Fetch current documents
        try:
            documents = self.notion.fetch_all_documents(self.config.notion_page_id)
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return [], [], []

        current_ids = {doc.id for doc in documents}
        cached_ids = set(self._document_hashes.keys())

        new_docs = []
        modified_docs = []
        deleted_ids = list(cached_ids - current_ids)

        for doc in documents:
            current_hash = self._compute_hash(doc)

            if doc.id not in self._document_hashes:
                # New document
                new_docs.append(doc)
                self._document_hashes[doc.id] = current_hash
            elif self._document_hashes[doc.id] != current_hash:
                # Modified document
                modified_docs.append(doc)
                self._document_hashes[doc.id] = current_hash

        # Remove deleted docs from cache
        for doc_id in deleted_ids:
            del self._document_hashes[doc_id]

        self._save_cache()

        # Log summary
        if new_docs or modified_docs or deleted_ids:
            print(f"  Changes detected:")
            if new_docs:
                print(f"    + {len(new_docs)} new: {[d.title for d in new_docs]}")
            if modified_docs:
                print(f"    ~ {len(modified_docs)} modified: {[d.title for d in modified_docs]}")
            if deleted_ids:
                print(f"    - {len(deleted_ids)} deleted")
        else:
            print("  No changes detected")

        return new_docs, modified_docs, deleted_ids

    def process_changes(
        self,
        new_docs: list[NotionPage],
        modified_docs: list[NotionPage],
        deleted_ids: list[str],
    ) -> int:
        """
        Process document changes by updating RAG embeddings.

        Returns:
            Number of chunks embedded
        """
        total_chunks = 0

        # Delete embeddings for removed/modified documents
        docs_to_clear = deleted_ids + [doc.id for doc in modified_docs]
        if docs_to_clear:
            print(f"  Clearing embeddings for {len(docs_to_clear)} documents...")
            for doc_id in docs_to_clear:
                self.rag.clear_embeddings_by_source(source_id=doc_id)

        # Embed new and modified documents
        docs_to_embed = new_docs + modified_docs
        if docs_to_embed:
            print(f"  Embedding {len(docs_to_embed)} documents...")
            total_chunks = self.rag.embed_notion_documents(docs_to_embed)
            print(f"  Embedded {total_chunks} chunks")

        return total_chunks

    def generate_posts_for_new_content(self, docs: list[NotionPage]):
        """
        Generate posts for new/modified documents (if auto_generate enabled).

        This sends posts to Telegram for HITL approval.
        """
        if not self.auto_generate or not docs:
            return

        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            print("  Telegram not configured, skipping auto-generation")
            return

        print(f"  Auto-generating posts for {len(docs)} documents...")

        from telegram_hitl import TelegramHITL
        from llm_client import LLMClient
        from prompts import SYSTEM_PROMPT, build_user_prompt

        llm = LLMClient(api_key=self.config.openrouter_api_key)

        for doc in docs:
            # Use the document title as the topic
            topic = doc.title

            # Get relevant context via RAG
            context = self.rag.search(topic, top_k=5)

            # Generate posts
            user_prompt = build_user_prompt(
                company_context=context,
                topic=topic,
                num_posts=3,
            )

            try:
                batch = llm.generate_posts(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )

                # Store posts for later approval
                for post in batch.posts:
                    post_id = self.db.create_post(
                        content=post.text,
                        topic=topic,
                        tone=post.tone.value,
                        source_doc_id=doc.id,
                    )
                    print(f"    Created post {post_id}: {post.text[:50]}...")

                print(f"  Generated {len(batch.posts)} posts for '{topic}'")

            except Exception as e:
                print(f"  Error generating posts for '{topic}': {e}")

    async def run_once(self) -> bool:
        """
        Run a single check for changes.

        Returns:
            True if changes were detected and processed
        """
        new_docs, modified_docs, deleted_ids = self.check_for_changes()

        if new_docs or modified_docs or deleted_ids:
            self.process_changes(new_docs, modified_docs, deleted_ids)

            if self.auto_generate:
                self.generate_posts_for_new_content(new_docs + modified_docs)

            return True

        return False

    async def run(self):
        """Run the watcher continuously."""
        print(f"Starting Notion watcher (polling every {self.poll_interval}s)")
        print(f"Auto-generate posts: {self.auto_generate}")
        print("Press Ctrl+C to stop\n")

        # Initial sync - embed all documents if cache is empty
        if not self._document_hashes:
            print("Initial sync: embedding all documents...")
            documents = self.notion.fetch_all_documents(self.config.notion_page_id)

            # Initialize vector table
            self.rag.init_vector_table()

            # Clear existing and embed fresh
            self.rag.clear_embeddings(source_type="notion_doc")
            total = self.rag.embed_notion_documents(documents)
            print(f"Initial sync complete: {total} chunks embedded\n")

            # Update cache
            for doc in documents:
                self._document_hashes[doc.id] = self._compute_hash(doc)
            self._save_cache()

        try:
            while True:
                await self.run_once()
                await asyncio.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\nStopping watcher...")


def add_clear_embeddings_by_source(rag_client_class):
    """Add method to RAGClient for clearing embeddings by source_id."""
    def clear_embeddings_by_source(self, source_id: str):
        """Clear embeddings for a specific source document."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get IDs to delete
        cursor.execute(
            "SELECT id FROM embeddings_meta WHERE source_id = ?",
            (source_id,)
        )
        ids = [row[0] for row in cursor.fetchall()]

        if ids:
            placeholders = ",".join("?" * len(ids))
            cursor.execute(f"DELETE FROM vec_embeddings WHERE rowid IN ({placeholders})", ids)
            cursor.execute(f"DELETE FROM embeddings_meta WHERE id IN ({placeholders})", ids)

        conn.commit()
        conn.close()

    rag_client_class.clear_embeddings_by_source = clear_embeddings_by_source


# Monkey-patch RAGClient with the new method
add_clear_embeddings_by_source(RAGClient)


async def main():
    parser = argparse.ArgumentParser(description="Watch Notion for document changes")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Polling interval in seconds (default: 300 = 5 min)"
    )
    parser.add_argument(
        "--auto-generate",
        action="store_true",
        help="Automatically generate posts for new/modified documents"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't poll continuously)"
    )

    args = parser.parse_args()

    config = Config.from_env()
    watcher = NotionWatcher(
        config=config,
        poll_interval=args.interval,
        auto_generate=args.auto_generate,
    )

    if args.once:
        changed = await watcher.run_once()
        print(f"\nChanges detected: {changed}")
    else:
        await watcher.run()


if __name__ == "__main__":
    asyncio.run(main())
