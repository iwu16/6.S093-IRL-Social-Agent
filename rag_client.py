"""
RAG (Retrieval-Augmented Generation) client for IRL Social Agent.

Handles:
- Document chunking
- Local embeddings via MiniLM-L6-v2 (fastembed)
- Vector storage in SQLite via sqlite-vec
- Hybrid search (BM25 keyword + semantic similarity)
"""

import json
import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sqlite_vec

# Handle both old and new fastembed API
try:
    from fastembed import TextEmbedding as Embedding
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Newer model for fastembed >= 0.2
except ImportError:
    from fastembed.embedding import FlagEmbedding as Embedding
    EMBEDDING_MODEL = "BAAI/bge-small-en"  # Older model for fastembed 0.1.x

from config import Config
from notion_client import NotionClient, NotionPage


# Embedding model configuration
EMBEDDING_DIM = 384  # BGE-small-en produces 384-dimensional vectors


@dataclass
class SearchResult:
    """A single search result with scores."""
    id: int
    content: str
    source_type: str
    source_id: Optional[str]
    metadata: dict
    bm25_score: float
    semantic_score: float
    final_score: float


class RAGClient:
    """RAG client for embedding, storing, and searching documents."""

    def __init__(self, db_path: Path = Path("irl_agent.db")):
        """
        Initialize RAG client.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._embedding_model: Optional[Embedding] = None

    @property
    def embedding_model(self) -> Embedding:
        """Lazy-load embedding model on first use."""
        if self._embedding_model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL}...")
            self._embedding_model = Embedding(model_name=EMBEDDING_MODEL)
            print("Embedding model loaded!")
        return self._embedding_model

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with sqlite-vec extension loaded."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def init_vector_table(self):
        """Initialize the vec_embeddings virtual table for vector search."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Create vector table using sqlite-vec
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                embedding float[{EMBEDDING_DIM}] distance_metric=cosine
            )
        """)

        conn.commit()
        conn.close()
        print("Vector table initialized!")

    # ==========================================
    # Document Chunking
    # ==========================================

    def chunk_document(
        self,
        content: str,
        title: str,
        chunk_by: str = "headers",
        max_chars: int = 500,
    ) -> list[dict]:
        """
        Split a document into chunks for embedding.

        Args:
            content: Document text content
            title: Document title
            chunk_by: Chunking strategy - "headers", "chars", or "paragraphs"
            max_chars: Max characters per chunk (for "chars" strategy)

        Returns:
            List of chunk dictionaries with content and metadata
        """
        if chunk_by == "headers":
            return self._chunk_by_headers(content, title)
        elif chunk_by == "chars":
            return self._chunk_by_chars(content, title, max_chars)
        elif chunk_by == "paragraphs":
            return self._chunk_by_paragraphs(content, title)
        else:
            # Default: treat entire document as one chunk
            return [{
                "content": f"# {title}\n\n{content}",
                "metadata": {"title": title, "section": "full"}
            }]

    def _chunk_by_headers(self, content: str, title: str) -> list[dict]:
        """Chunk by ## headers (markdown sections)."""
        # Split on ## headers
        sections = re.split(r'(?=^##\s+)', content, flags=re.MULTILINE)

        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Extract section title if present
            section_match = re.search(r'^##\s+(.+)$', section, re.MULTILINE)
            section_title = section_match.group(1) if section_match else "Introduction"

            # Build chunk with document context
            chunk_content = f"# {title}\n\n{section}"

            chunks.append({
                "content": chunk_content,
                "metadata": {
                    "title": title,
                    "section": section_title,
                }
            })

        # If no sections found, return whole document
        if not chunks:
            chunks.append({
                "content": f"# {title}\n\n{content}",
                "metadata": {"title": title, "section": "full"}
            })

        return chunks

    def _chunk_by_chars(self, content: str, title: str, max_chars: int) -> list[dict]:
        """Chunk by fixed character count with overlap."""
        chunks = []
        overlap = 50  # Characters of overlap between chunks

        start = 0
        chunk_num = 1
        while start < len(content):
            end = start + max_chars

            # Try to break at a sentence boundary
            if end < len(content):
                # Look for sentence end near the boundary
                for punct in ['. ', '! ', '? ', '\n\n']:
                    last_punct = content[start:end].rfind(punct)
                    if last_punct > max_chars * 0.5:  # At least halfway through
                        end = start + last_punct + len(punct)
                        break

            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append({
                    "content": f"# {title} (Part {chunk_num})\n\n{chunk_text}",
                    "metadata": {
                        "title": title,
                        "section": f"part_{chunk_num}",
                        "char_start": start,
                        "char_end": end,
                    }
                })
                chunk_num += 1

            start = end - overlap if end < len(content) else end

        return chunks if chunks else [{"content": content, "metadata": {"title": title}}]

    def _chunk_by_paragraphs(self, content: str, title: str) -> list[dict]:
        """Chunk by paragraph boundaries."""
        paragraphs = content.split('\n\n')
        chunks = []

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para and len(para) > 20:  # Skip very short paragraphs
                chunks.append({
                    "content": f"# {title}\n\n{para}",
                    "metadata": {
                        "title": title,
                        "section": f"paragraph_{i + 1}",
                    }
                })

        return chunks if chunks else [{"content": content, "metadata": {"title": title}}]

    # ==========================================
    # Embedding
    # ==========================================

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = list(self.embedding_model.embed([text]))
        return embeddings[0].tolist()

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (more efficient)."""
        if not texts:
            return []
        embeddings = list(self.embedding_model.embed(texts))
        return [emb.tolist() for emb in embeddings]

    @staticmethod
    def serialize_embedding(embedding: list[float]) -> bytes:
        """Serialize embedding to binary format for sqlite-vec."""
        return struct.pack(f'{len(embedding)}f', *embedding)

    # ==========================================
    # Storage
    # ==========================================

    def save_embedding(
        self,
        source_type: str,
        content: str,
        embedding: list[float],
        source_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Save an embedding to the database.

        Args:
            source_type: Type of source (e.g., 'notion_doc', 'generated_post')
            content: The text content
            embedding: The embedding vector
            source_id: Optional source identifier
            metadata: Optional metadata dict

        Returns:
            The rowid of the saved embedding
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Insert metadata (FTS5 index updated via trigger)
        cursor.execute(
            """
            INSERT INTO embeddings_meta (source_type, source_id, content, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (source_type, source_id, content, json.dumps(metadata) if metadata else None),
        )
        rowid = cursor.lastrowid

        # Insert vector with matching rowid
        cursor.execute(
            """
            INSERT INTO vec_embeddings (rowid, embedding)
            VALUES (?, ?)
            """,
            (rowid, self.serialize_embedding(embedding)),
        )

        conn.commit()
        conn.close()
        return rowid

    def embed_notion_documents(
        self,
        documents: list[NotionPage],
        chunk_by: str = "headers",
    ) -> int:
        """
        Embed and store Notion documents.

        Args:
            documents: List of NotionPage objects
            chunk_by: Chunking strategy

        Returns:
            Number of chunks embedded
        """
        total_chunks = 0

        for doc in documents:
            print(f"Processing: {doc.title}")

            # Chunk the document
            chunks = self.chunk_document(doc.content, doc.title, chunk_by=chunk_by)

            # Generate embeddings in batch
            texts = [c["content"] for c in chunks]
            embeddings = self.generate_embeddings_batch(texts)

            # Save each chunk
            for chunk, embedding in zip(chunks, embeddings):
                self.save_embedding(
                    source_type="notion_doc",
                    content=chunk["content"],
                    embedding=embedding,
                    source_id=doc.id,
                    metadata=chunk["metadata"],
                )

            print(f"  Saved {len(chunks)} chunks")
            total_chunks += len(chunks)

        return total_chunks

    def embed_generated_post(
        self,
        post_text: str,
        post_id: int,
        topic: str,
        tone: Optional[str] = None,
    ) -> int:
        """
        Embed a generated post into the RAG database.

        This allows the system to:
        - Avoid generating duplicate posts
        - Reference past posts for consistency
        - Build a knowledge base of generated content

        Args:
            post_text: The post content
            post_id: Database ID of the post
            topic: Topic the post was generated for
            tone: Tone of the post

        Returns:
            The rowid of the saved embedding
        """
        # Generate embedding for the post
        embedding = self.generate_embedding(post_text)

        # Save with metadata
        return self.save_embedding(
            source_type="generated_post",
            content=post_text,
            embedding=embedding,
            source_id=str(post_id),
            metadata={
                "topic": topic,
                "tone": tone,
                "type": "social_post",
            },
        )

    def find_similar_posts(self, text: str, threshold: float = 0.8, limit: int = 5) -> list[dict]:
        """
        Find posts similar to the given text.

        Useful for detecting duplicates before posting.

        Args:
            text: Text to compare against
            threshold: Similarity threshold (0-1, higher = more similar)
            limit: Max results to return

        Returns:
            List of similar posts with similarity scores
        """
        embedding = self.generate_embedding(text)
        distances = self.semantic_search(embedding, limit=limit * 2)

        conn = self.get_connection()
        cursor = conn.cursor()

        similar = []
        for rowid, distance in distances.items():
            # Convert distance to similarity (0=identical, 2=opposite)
            similarity = 1 - (distance / 2)

            if similarity >= threshold:
                cursor.execute(
                    "SELECT content, source_type, source_id, metadata FROM embeddings_meta WHERE id = ?",
                    (rowid,)
                )
                row = cursor.fetchone()
                if row and row[1] == "generated_post":
                    similar.append({
                        "content": row[0],
                        "source_id": row[2],
                        "metadata": json.loads(row[3]) if row[3] else {},
                        "similarity": similarity,
                    })

        conn.close()
        return similar[:limit]

    def clear_embeddings(self, source_type: Optional[str] = None):
        """Clear embeddings from the database."""
        conn = self.get_connection()
        cursor = conn.cursor()

        if source_type:
            # Get IDs to delete
            cursor.execute(
                "SELECT id FROM embeddings_meta WHERE source_type = ?",
                (source_type,)
            )
            ids = [row[0] for row in cursor.fetchall()]

            if ids:
                placeholders = ",".join("?" * len(ids))
                cursor.execute(f"DELETE FROM vec_embeddings WHERE rowid IN ({placeholders})", ids)
                cursor.execute(f"DELETE FROM embeddings_meta WHERE id IN ({placeholders})", ids)
        else:
            cursor.execute("DELETE FROM vec_embeddings")
            cursor.execute("DELETE FROM embeddings_meta")

        conn.commit()
        conn.close()

    # ==========================================
    # Search
    # ==========================================

    def bm25_search(self, query: str, limit: int = 100) -> dict[int, float]:
        """
        Search using BM25 ranking via FTS5.

        Returns dict mapping embedding_id to raw BM25 score.
        Note: FTS5 BM25 scores are NEGATIVE (more negative = better match).
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Escape special FTS5 characters
        safe_query = query.replace('"', '""')

        try:
            cursor.execute("""
                SELECT rowid, bm25(embeddings_fts) as score
                FROM embeddings_fts
                WHERE embeddings_fts MATCH ?
                LIMIT ?
            """, (safe_query, limit))

            results = {row[0]: row[1] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            results = {}

        conn.close()
        return results

    def semantic_search(self, query_embedding: list[float], limit: int = 100) -> dict[int, float]:
        """
        Search using sqlite-vec cosine distance.

        Returns dict mapping rowid to cosine distance.
        Note: distance is in [0, 2] where 0 = identical, 2 = opposite.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT rowid, distance
            FROM vec_embeddings
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance
        """, (self.serialize_embedding(query_embedding), limit))

        results = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return results

    def _normalize_bm25_scores(self, scores: dict[int, float]) -> dict[int, float]:
        """Normalize BM25 scores to [0, 1] range."""
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)  # Most negative = best
        max_score = max(values)  # Least negative = worst

        if min_score == max_score:
            return {id: 1.0 for id in scores}

        score_range = max_score - min_score
        return {
            id: (max_score - score) / score_range
            for id, score in scores.items()
        }

    def _normalize_distances(self, distances: dict[int, float]) -> dict[int, float]:
        """Normalize cosine distances to similarity scores in [0, 1]."""
        if not distances:
            return {}

        # Convert distances to similarities
        similarities = {id: 1 - (dist / 2) for id, dist in distances.items()}

        min_sim = min(similarities.values())
        max_sim = max(similarities.values())

        if min_sim == max_sim:
            return {id: 1.0 for id in similarities}

        sim_range = max_sim - min_sim
        return {
            id: (sim - min_sim) / sim_range
            for id, sim in similarities.items()
        }

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining BM25 and semantic similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            keyword_weight: Weight for BM25 (0-1)
            semantic_weight: Weight for semantic similarity (0-1)

        Returns:
            List of SearchResult objects sorted by combined score
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Get BM25 scores
        bm25_raw = self.bm25_search(query)
        bm25_normalized = self._normalize_bm25_scores(bm25_raw)

        # Get semantic distances
        semantic_raw = self.semantic_search(query_embedding)
        semantic_normalized = self._normalize_distances(semantic_raw)

        # Combine all unique IDs
        all_ids = set(bm25_normalized.keys()) | set(semantic_normalized.keys())

        if not all_ids:
            return []

        # Get metadata for all candidates
        conn = self.get_connection()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(all_ids))
        cursor.execute(f"""
            SELECT id, source_type, source_id, content, metadata
            FROM embeddings_meta
            WHERE id IN ({placeholders})
        """, list(all_ids))

        metadata_map = {}
        for row in cursor.fetchall():
            metadata_map[row[0]] = {
                "source_type": row[1],
                "source_id": row[2],
                "content": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
            }
        conn.close()

        # Compute combined scores
        results = []
        for id in all_ids:
            bm25_score = bm25_normalized.get(id, 0.0)
            semantic_score = semantic_normalized.get(id, 0.0)
            final_score = (keyword_weight * bm25_score) + (semantic_weight * semantic_score)

            meta = metadata_map.get(id, {})
            results.append(SearchResult(
                id=id,
                content=meta.get("content", ""),
                source_type=meta.get("source_type", ""),
                source_id=meta.get("source_id"),
                metadata=meta.get("metadata", {}),
                bm25_score=bm25_score,
                semantic_score=semantic_score,
                final_score=final_score,
            ))

        # Sort by final score (descending)
        results.sort(key=lambda x: x.final_score, reverse=True)

        return results[:top_k]

    def search(self, query: str, top_k: int = 5) -> str:
        """
        Search for relevant context and return formatted string.

        This is the main entry point for RAG retrieval.

        Args:
            query: Search query (e.g., topic for post generation)
            top_k: Number of results to include

        Returns:
            Formatted context string for LLM prompt
        """
        results = self.hybrid_search(query, top_k=top_k)

        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            section = result.metadata.get("section", "unknown")
            score = result.final_score
            context_parts.append(
                f"[{i}. {result.source_type} - {section}] (relevance: {score:.2f})\n"
                f"{result.content}"
            )

        return "\n\n---\n\n".join(context_parts)


# ==========================================
# CLI for testing and embedding
# ==========================================

def embed_from_notion(config: Config, chunk_by: str = "headers"):
    """Fetch Notion docs and embed them."""
    print("Fetching documents from Notion...")
    notion = NotionClient(token=config.notion_token)
    documents = notion.fetch_all_documents(config.notion_page_id)
    print(f"Retrieved {len(documents)} documents")

    rag = RAGClient()
    rag.init_vector_table()

    # Clear existing notion embeddings
    print("Clearing existing Notion embeddings...")
    rag.clear_embeddings(source_type="notion_doc")

    # Embed new documents
    print(f"Embedding documents (chunking by: {chunk_by})...")
    total = rag.embed_notion_documents(documents, chunk_by=chunk_by)
    print(f"\nTotal chunks embedded: {total}")

    return total


def test_search(query: str, top_k: int = 5):
    """Test RAG search."""
    rag = RAGClient()
    print(f"\nSearching for: '{query}'\n")

    results = rag.hybrid_search(query, top_k=top_k)

    for i, r in enumerate(results, 1):
        print(f"{i}. Score: {r.final_score:.3f} (BM25: {r.bm25_score:.3f}, Semantic: {r.semantic_score:.3f})")
        print(f"   Section: {r.metadata.get('section', 'N/A')}")
        print(f"   Preview: {r.content[:150]}...\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Client CLI")
    parser.add_argument("--embed", action="store_true", help="Embed Notion documents")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--chunk-by", type=str, default="headers",
                        choices=["headers", "chars", "paragraphs"],
                        help="Chunking strategy")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if args.embed:
        config = Config.from_env()
        embed_from_notion(config, chunk_by=args.chunk_by)
    elif args.search:
        test_search(args.search, top_k=args.top_k)
    else:
        print("Usage:")
        print("  python rag_client.py --embed              # Embed Notion docs")
        print("  python rag_client.py --search 'query'     # Test search")
