"""
IRL Social Media Agent - Main Entry Point

A prototype agent that:
1. Fetches company documents from Notion
2. Uses them as context for an LLM
3. Generates structured social media posts
4. Posts to Mastodon with interactive approval
"""

import argparse
import json
from config import Config
from notion_client import NotionClient, format_documents_as_context
from llm_client import LLMClient
from mastodon_client import MastodonClient
from prompts import SYSTEM_PROMPT, build_user_prompt, build_reply_prompt
from schemas import PostBatch, SocialPost, ReplyBatch, Reply, SAMPLE_RESPONSE


def fetch_company_context(config: Config) -> str:
    """
    Fetch and format company documents from Notion.

    Returns:
        Formatted context string for LLM
    """
    print("Fetching company documents from Notion...")

    client = NotionClient(token=config.notion_token)
    documents = client.fetch_all_documents(config.notion_page_id)

    print(f"  Retrieved {len(documents)} documents")
    for doc in documents:
        print(f"    - {doc.title}")

    return format_documents_as_context(documents)


def generate_posts(config: Config, context: str, topic: str, num_posts: int = 5) -> PostBatch:
    """
    Generate social media posts using LLM with company context.

    Args:
        config: Application configuration
        context: Formatted company documents
        topic: Topic for post generation
        num_posts: Number of posts to generate

    Returns:
        Validated PostBatch with generated posts
    """
    print(f"\nGenerating {num_posts} posts about: '{topic}'...")

    client = LLMClient(api_key=config.openrouter_api_key)

    user_prompt = build_user_prompt(
        company_context=context,
        topic=topic,
        num_posts=num_posts
    )

    return client.generate_posts(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt
    )


def display_post(post: SocialPost, index: int) -> None:
    """Display a single post for review."""
    print(f"\n--- Post {index} ---")
    print(f"Text: {post.text}")
    print(f"Tone: {post.tone.value}")
    print(f"Use:  {post.intended_use.value}")
    print(f"Chars: {len(post.text)}/500")


def display_posts_for_review(batch: PostBatch) -> None:
    """
    Display generated posts for human review.

    Args:
        batch: PostBatch containing generated posts
    """
    print("\n" + "=" * 60)
    print("GENERATED POSTS FOR REVIEW")
    print(f"Topic: {batch.topic}")
    print("=" * 60)

    for i, post in enumerate(batch.posts, 1):
        display_post(post, i)

    print("\n" + "=" * 60)
    print("Review complete.")
    print("=" * 60)


def post_to_mastodon_interactive(config: Config, batch: PostBatch) -> None:
    """
    Interactively approve and post each generated post to Mastodon.

    Args:
        config: Application configuration
        batch: PostBatch containing generated posts
    """
    if not config.mastodon_token:
        print("\nNo Mastodon token configured. Set MASTODON_TOKEN in .env to enable posting.")
        return

    client = MastodonClient(
        token=config.mastodon_token,
        instance=config.mastodon_instance
    )

    # Verify credentials first
    print(f"\nConnecting to {config.mastodon_instance}...")
    if not client.verify_credentials():
        print("Failed to verify Mastodon credentials. Check your token.")
        return

    print("Connected successfully!\n")

    posted_count = 0
    skipped_count = 0

    for i, post in enumerate(batch.posts, 1):
        print("\n" + "-" * 40)
        display_post(post, i)
        print("-" * 40)

        while True:
            choice = input("\nPost this to Mastodon? [y]es / [n]o / [q]uit: ").strip().lower()

            if choice in ("y", "yes"):
                print("Posting...")
                result = client.post_status(post.text)

                if result.success:
                    print(f"Posted! {result.url}")
                    posted_count += 1
                else:
                    print(f"Failed to post: {result.error}")
                break

            elif choice in ("n", "no"):
                print("Skipped.")
                skipped_count += 1
                break

            elif choice in ("q", "quit"):
                print("\nStopping. Remaining posts not reviewed.")
                print(f"\nSummary: {posted_count} posted, {skipped_count} skipped")
                return

            else:
                print("Please enter 'y', 'n', or 'q'")

    print(f"\nAll done! {posted_count} posted, {skipped_count} skipped")


def export_as_json(batch: PostBatch, filename: str = "output.json") -> None:
    """Export posts to JSON file for further processing."""
    with open(filename, "w") as f:
        json.dump(batch.model_dump(), f, indent=2)
    print(f"\nExported to {filename}")


def search_and_reply(config: Config, context: str, keyword: str) -> None:
    """
    Search for posts by keyword and generate/post replies.

    Args:
        config: Application configuration
        context: Formatted company documents
        keyword: Keyword to search for
    """
    if not config.mastodon_token:
        print("\nNo Mastodon token configured. Set MASTODON_TOKEN in .env")
        return

    mastodon = MastodonClient(
        token=config.mastodon_token,
        instance=config.mastodon_instance
    )

    # Step 1: Search for posts by hashtag
    print(f"\nSearching Mastodon for posts with #{keyword.lstrip('#')}...")
    posts = mastodon.search_posts(keyword, limit=5)

    if not posts:
        print("No posts found for that keyword.")
        return

    print(f"Found {len(posts)} posts:\n")
    for i, post in enumerate(posts, 1):
        print(f"--- Post {i} ---")
        print(f"Author: {post.author} ({post.author_handle})")
        print(f"Content: {post.content[:200]}{'...' if len(post.content) > 200 else ''}")
        print(f"URL: {post.url}")
        print()

    # Step 2: Generate replies with LLM (structured output for all at once)
    print("Generating replies with LLM...")
    llm = LLMClient(api_key=config.openrouter_api_key)

    reply_prompt = build_reply_prompt(
        company_context=context,
        keyword=keyword,
        posts=posts,
    )

    reply_batch = llm.generate_replies(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=reply_prompt
    )

    # Step 3: Display and interactively post replies
    print("\n" + "=" * 60)
    print("GENERATED REPLIES")
    print("=" * 60)

    # Create a lookup for original posts
    post_lookup = {p.id: p for p in posts}

    posted_count = 0
    skipped_count = 0

    for i, reply in enumerate(reply_batch.replies, 1):
        original = post_lookup.get(reply.original_post_id)

        print("\n" + "-" * 40)
        print(f"Reply {i}")
        if original:
            print(f"To: {original.author} ({original.author_handle})")
            print(f"Original: {original.content[:100]}...")
        print(f"Reply: {reply.reply_text}")
        print(f"Tone: {reply.tone.value}")
        print(f"Chars: {len(reply.reply_text)}/500")
        print("-" * 40)

        while True:
            choice = input("\nPost this reply? [y]es / [n]o / [q]uit: ").strip().lower()

            if choice in ("y", "yes"):
                print("Posting reply...")
                result = mastodon.reply_to_post(
                    text=reply.reply_text,
                    reply_to_id=reply.original_post_id
                )

                if result.success:
                    print(f"Replied! {result.url}")
                    posted_count += 1
                else:
                    print(f"Failed: {result.error}")
                break

            elif choice in ("n", "no"):
                print("Skipped.")
                skipped_count += 1
                break

            elif choice in ("q", "quit"):
                print("\nStopping.")
                print(f"\nSummary: {posted_count} replied, {skipped_count} skipped")
                return

            else:
                print("Please enter 'y', 'n', or 'q'")

    print(f"\nAll done! {posted_count} replied, {skipped_count} skipped")


def run_demo_mode() -> None:
    """Run with sample data (no API calls) for demo/testing."""
    print("\n[DEMO MODE - Using sample data, no API calls]\n")

    batch = PostBatch.model_validate(SAMPLE_RESPONSE)
    display_posts_for_review(batch)
    export_as_json(batch, "demo_output.json")


def main():
    parser = argparse.ArgumentParser(
        description="Generate IRL social media posts using company context from Notion"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="spontaneous plans",
        help="Topic for post generation (e.g., 'meeting people without pressure')"
    )
    parser.add_argument(
        "--num-posts",
        type=int,
        default=5,
        choices=[3, 4, 5],
        help="Number of posts to generate (3-5)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with sample data (no API calls)"
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILE",
        help="Export posts to JSON file"
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Interactively post approved content to Mastodon"
    )
    parser.add_argument(
        "--reply",
        type=str,
        metavar="KEYWORD",
        help="Search for posts by keyword and generate replies"
    )

    args = parser.parse_args()

    # Demo mode for testing without API keys
    if args.demo:
        run_demo_mode()
        return

    # Production mode
    try:
        # Load configuration
        config = Config.from_env()

        # Step 1: Fetch company context from Notion
        context = fetch_company_context(config)

        # Step 2: Generate posts with LLM
        batch = generate_posts(
            config=config,
            context=context,
            topic=args.topic,
            num_posts=args.num_posts
        )

        # Step 3: Display for human review
        display_posts_for_review(batch)

        # Step 4: Optional export
        if args.export:
            export_as_json(batch, args.export)

        # Step 5: Interactive posting to Mastodon
        if args.post:
            post_to_mastodon_interactive(config, batch)

        # Step 6: Reply mode - search and reply to posts
        if args.reply:
            search_and_reply(config, context, args.reply)

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Run with --demo flag to test without API keys")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
