"""
SQLite database module for IRL Social Agent.

Handles all database operations for posts, approvals, and analytics.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default database path
DEFAULT_DB_PATH = Path(__file__).parent / "irl_agent.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass
class PostRecord:
    """Database record for a post."""
    id: int
    content: str
    topic: Optional[str]
    tone: Optional[str]
    source_doc_id: Optional[str]
    character_count: int
    created_at: datetime
    updated_at: datetime


@dataclass
class ApprovalRecord:
    """Database record for an approval decision."""
    id: int
    post_id: int
    decision: str
    edited_content: Optional[str]
    rejection_reason: Optional[str]
    approved_by: Optional[str]
    approved_at: datetime


@dataclass
class PublishedPostRecord:
    """Database record for a published post."""
    id: int
    post_id: int
    approval_id: Optional[int]
    mastodon_status_id: Optional[str]
    mastodon_url: Optional[str]
    published_at: datetime


class Database:
    """SQLite database manager for IRL Social Agent."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Uses default if not provided.
        """
        self.db_path = db_path or DEFAULT_DB_PATH

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        """Initialize database with schema."""
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

        with open(SCHEMA_PATH) as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)
        print(f"Database initialized at: {self.db_path}")

    # ==========================================
    # Post Operations
    # ==========================================

    def create_post(
        self,
        content: str,
        topic: Optional[str] = None,
        tone: Optional[str] = None,
        source_doc_id: Optional[str] = None,
    ) -> int:
        """
        Create a new post record.

        Returns:
            The ID of the created post.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO posts (content, topic, tone, source_doc_id, character_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content, topic, tone, source_doc_id, len(content)),
            )
            return cursor.lastrowid

    def get_post(self, post_id: int) -> Optional[PostRecord]:
        """Get a post by ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM posts WHERE id = ?", (post_id,)
            ).fetchone()
            if row:
                return PostRecord(**dict(row))
            return None

    def get_recent_posts(self, limit: int = 10) -> list[PostRecord]:
        """Get recent posts."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM posts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [PostRecord(**dict(row)) for row in rows]

    # ==========================================
    # Approval Operations
    # ==========================================

    def record_approval(
        self,
        post_id: int,
        decision: str,
        edited_content: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        approved_by: Optional[str] = None,
    ) -> int:
        """
        Record an approval decision.

        Returns:
            The ID of the created approval record.
        """
        if decision not in ("approve", "reject", "edit"):
            raise ValueError(f"Invalid decision: {decision}")

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO approvals
                (post_id, decision, edited_content, rejection_reason, approved_by)
                VALUES (?, ?, ?, ?, ?)
                """,
                (post_id, decision, edited_content, rejection_reason, approved_by),
            )
            return cursor.lastrowid

    def get_approval_stats(self) -> dict:
        """Get approval statistics."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT decision, COUNT(*) as count
                FROM approvals
                GROUP BY decision
                """
            ).fetchall()
            return {row["decision"]: row["count"] for row in rows}

    # ==========================================
    # Published Post Operations
    # ==========================================

    def record_published_post(
        self,
        post_id: int,
        mastodon_status_id: Optional[str] = None,
        mastodon_url: Optional[str] = None,
        approval_id: Optional[int] = None,
    ) -> int:
        """
        Record a published post.

        Returns:
            The ID of the created published post record.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO published_posts
                (post_id, approval_id, mastodon_status_id, mastodon_url)
                VALUES (?, ?, ?, ?)
                """,
                (post_id, approval_id, mastodon_status_id, mastodon_url),
            )
            return cursor.lastrowid

    def get_published_posts(self, limit: int = 10) -> list[PublishedPostRecord]:
        """Get recent published posts."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM published_posts ORDER BY published_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [PublishedPostRecord(**dict(row)) for row in rows]

    # ==========================================
    # Image Operations
    # ==========================================

    def record_generated_image(
        self,
        image_url: str,
        prompt: str,
        post_id: Optional[int] = None,
        model_version: Optional[str] = None,
        generation_time_seconds: Optional[float] = None,
    ) -> int:
        """Record a generated image."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO generated_images
                (post_id, prompt, image_url, model_version, generation_time_seconds)
                VALUES (?, ?, ?, ?, ?)
                """,
                (post_id, prompt, image_url, model_version, generation_time_seconds),
            )
            return cursor.lastrowid

    # ==========================================
    # Notion Document Operations
    # ==========================================

    def cache_notion_document(
        self,
        notion_page_id: str,
        title: str,
        content: str,
    ) -> int:
        """Cache a Notion document."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO notion_documents (notion_page_id, title, content)
                VALUES (?, ?, ?)
                ON CONFLICT(notion_page_id) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (notion_page_id, title, content),
            )
            return cursor.lastrowid

    def get_cached_document(self, notion_page_id: str) -> Optional[dict]:
        """Get a cached Notion document."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM notion_documents WHERE notion_page_id = ?",
                (notion_page_id,),
            ).fetchone()
            if row:
                return dict(row)
            return None


def init_database(db_path: Optional[Path] = None):
    """Initialize the database with schema."""
    db = Database(db_path)
    db.init_db()
    return db


def init_database_with_rag(db_path: Optional[Path] = None):
    """Initialize database with schema and RAG vector table."""
    db = Database(db_path)
    db.init_db()

    # Initialize RAG vector table (requires sqlite-vec)
    try:
        import sqlite_vec
        conn = sqlite3.connect(db.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                embedding float[384] distance_metric=cosine
            )
        """)
        conn.commit()
        conn.close()
        print("RAG vector table initialized!")
    except ImportError:
        print("Warning: sqlite-vec not installed. RAG features unavailable.")
    except Exception as e:
        print(f"Warning: Could not initialize vector table: {e}")

    return db


if __name__ == "__main__":
    # Initialize database when run directly
    db = init_database_with_rag()
    print("Database schema created successfully!")

    # Show tables
    with db.get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        print(f"\nTables: {[t['name'] for t in tables]}")
