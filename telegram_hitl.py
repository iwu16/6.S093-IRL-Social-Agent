"""
Telegram Human-in-the-Loop (HITL) approval workflow.

Sends posts to Telegram for human approval before posting to Mastodon.
Supports: Approve, Reject (with reason), and Edit actions.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Optional
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, MessageHandler, filters, ContextTypes


@dataclass
class ApprovalResult:
    """Result of a human approval decision."""
    decision: str  # "approve", "reject", or "edit"
    edited_text: Optional[str] = None
    rejection_reason: Optional[str] = None


class TelegramHITL:
    """Human-in-the-Loop approval via Telegram."""

    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram HITL client.

        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID
        """
        self.bot_token = bot_token
        self.chat_id = int(chat_id)

        # State for the approval flow
        self._pending_post: Optional[str] = None
        self._decision: Optional[str] = None
        self._edited_text: Optional[str] = None
        self._rejection_reason: Optional[str] = None
        self._waiting_for_input: Optional[str] = None  # "edit" or "reason"
        self._done_event: Optional[asyncio.Event] = None

    async def request_approval(
        self,
        post_text: str,
        post_index: int = 1,
        total_posts: int = 1,
        topic: str = "",
        tone: str = "",
    ) -> ApprovalResult:
        """
        Send a post to Telegram for human approval.

        Args:
            post_text: The post content to approve
            post_index: Which post number this is
            total_posts: Total number of posts in batch
            topic: The topic of the post
            tone: The tone of the post

        Returns:
            ApprovalResult with decision and optional edits/feedback
        """
        # Reset state
        self._pending_post = post_text
        self._decision = None
        self._edited_text = None
        self._rejection_reason = None
        self._waiting_for_input = None
        self._done_event = asyncio.Event()

        # Build the message
        message_text = (
            f"📝 Post {post_index}/{total_posts} for Approval\n\n"
            f"{post_text}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Characters: {len(post_text)}/500\n"
        )
        if topic:
            message_text += f"🏷️ Topic: {topic}\n"
        if tone:
            message_text += f"🎭 Tone: {tone}\n"

        # Create keyboard with Approve/Edit/Reject buttons
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("✏️ Edit", callback_data="edit"),
                InlineKeyboardButton("❌ Reject", callback_data="reject"),
            ]
        ])

        # Send the message
        bot = Bot(token=self.bot_token)
        await bot.send_message(
            chat_id=self.chat_id,
            text=message_text,
            reply_markup=keyboard,
        )
        print("📱 Sent to Telegram. Waiting for approval...")

        # Set up the application and handlers
        app = Application.builder().token(self.bot_token).build()
        app.add_handler(CallbackQueryHandler(self._handle_button))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))

        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        # Wait for decision
        await self._done_event.wait()

        # Cleanup
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

        return ApprovalResult(
            decision=self._decision,
            edited_text=self._edited_text,
            rejection_reason=self._rejection_reason,
        )

    async def _handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button clicks from Telegram."""
        query = update.callback_query
        await query.answer()

        action = query.data

        if action == "approve":
            self._decision = "approve"
            await query.edit_message_text(
                f"✅ APPROVED\n\n{self._pending_post}"
            )
            self._done_event.set()

        elif action == "edit":
            self._waiting_for_input = "edit"
            await query.edit_message_text(
                f"✏️ EDITING\n\n"
                f"Current text:\n{self._pending_post}\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Reply with your edited version of the post."
            )

        elif action == "reject":
            self._waiting_for_input = "reason"
            await query.edit_message_text(
                f"❌ REJECTING\n\n"
                f"Original:\n{self._pending_post[:100]}...\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Reply with the reason for rejection.\n"
                f"(This helps improve future posts)"
            )

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (edits or rejection reasons)."""
        if not self._waiting_for_input:
            return

        text = update.message.text

        if self._waiting_for_input == "edit":
            self._decision = "edit"
            self._edited_text = text
            await update.message.reply_text(
                f"✏️ Edit received!\n\n"
                f"New text:\n{text}\n\n"
                f"Characters: {len(text)}/500"
            )
            self._waiting_for_input = None
            self._done_event.set()

        elif self._waiting_for_input == "reason":
            self._decision = "reject"
            self._rejection_reason = text
            await update.message.reply_text(
                f"📝 Feedback recorded!\n\n"
                f"Reason: {text}"
            )
            self._waiting_for_input = None
            self._done_event.set()

    async def send_notification(self, message: str):
        """Send a simple notification message to Telegram."""
        bot = Bot(token=self.bot_token)
        await bot.send_message(chat_id=self.chat_id, text=message)


async def test_hitl():
    """Test the HITL workflow."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your environment")
        return

    hitl = TelegramHITL(bot_token, chat_id)

    sample_post = (
        "Sometimes the best moments happen when you least expect them. "
        "A coffee with someone new. A walk that turns into a real conversation. "
        "That's what IRL is about."
    )

    result = await hitl.request_approval(
        post_text=sample_post,
        post_index=1,
        total_posts=3,
        topic="spontaneous connections",
        tone="warm",
    )

    print(f"\n📊 Result:")
    print(f"   Decision: {result.decision}")
    if result.edited_text:
        print(f"   Edited text: {result.edited_text}")
    if result.rejection_reason:
        print(f"   Rejection reason: {result.rejection_reason}")


if __name__ == "__main__":
    asyncio.run(test_hitl())
