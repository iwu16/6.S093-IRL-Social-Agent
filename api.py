"""
FastAPI application for IRL Social Media Agent.

Provides REST API endpoints for:
- Post generation via LLM
- Database operations (posts, approvals)
- Mastodon posting
- Health checks
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import Config
from database import Database
from schemas import PostBatch, PostTone, IntendedUse


# ===========================================
# API Request/Response Models
# ===========================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    timestamp: str


class GeneratePostsRequest(BaseModel):
    """Request to generate posts."""
    topic: str = Field(..., description="Topic for post generation")
    num_posts: int = Field(default=5, ge=3, le=5, description="Number of posts (3-5)")
    embed_posts: bool = Field(default=False, description="Embed generated posts into RAG for duplicate detection")
    with_images: bool = Field(default=False, description="Generate mascot images for each post")


class PostResponse(BaseModel):
    """Single post from database."""
    id: int
    content: str
    topic: Optional[str]
    tone: Optional[str]
    character_count: int
    created_at: str


class ApprovalRequest(BaseModel):
    """Request to record an approval decision."""
    decision: str = Field(..., pattern="^(approve|reject|edit)$")
    edited_content: Optional[str] = None
    rejection_reason: Optional[str] = None


class ApprovalResponse(BaseModel):
    """Approval record response."""
    id: int
    post_id: int
    decision: str
    approved_at: str


class StatsResponse(BaseModel):
    """Approval statistics response."""
    total_posts: int
    approval_stats: dict
    recent_posts_count: int


class MastodonPostRequest(BaseModel):
    """Request to post to Mastodon."""
    text: str = Field(..., max_length=500)
    post_id: Optional[int] = None  # Optional link to DB post


class MastodonPostResponse(BaseModel):
    """Mastodon post result."""
    success: bool
    url: Optional[str] = None
    error: Optional[str] = None


class RAGEmbedRequest(BaseModel):
    """Request to embed documents into RAG."""
    # Empty for now - will fetch from Notion


class RAGEmbedResponse(BaseModel):
    """RAG embedding result."""
    success: bool
    chunks_embedded: int
    documents_processed: int
    message: str


class RAGSearchRequest(BaseModel):
    """Request to search RAG database."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")


class RAGSearchResult(BaseModel):
    """Single search result."""
    content: str
    source_type: str
    source_id: Optional[str]
    score: float


class RAGSearchResponse(BaseModel):
    """RAG search results."""
    query: str
    results: list[RAGSearchResult]
    total_results: int


class GeneratedImage(BaseModel):
    """A generated image."""
    image_url: str
    post_text: str


class ImageGenerationRequest(BaseModel):
    """Request to generate a mascot image."""
    prompt: Optional[str] = Field(None, description="Custom prompt (optional)")
    post_text: Optional[str] = Field(None, description="Post text to base image on")
    aspect_ratio: str = Field(default="1:1", description="Image aspect ratio")


class ImageGenerationResponse(BaseModel):
    """Image generation result."""
    success: bool
    image_url: Optional[str] = None
    error: Optional[str] = None


class PostWithImageResponse(BaseModel):
    """A generated post with optional image."""
    text: str
    tone: str
    intended_use: str
    image_url: Optional[str] = None


class PostBatchWithImages(BaseModel):
    """Batch of posts with optional images."""
    topic: str
    posts: list[PostWithImageResponse]


# ===========================================
# Application Setup
# ===========================================

# Global database instance
db: Optional[Database] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    global db
    db = Database()
    db.init_db()
    print("Database initialized")
    yield
    print("Shutting down")


app = FastAPI(
    title="IRL Social Media Agent API",
    description="REST API for generating and managing social media posts",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================
# Health Endpoints
# ===========================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - basic health check."""
    return HealthResponse(
        status="ok",
        database="connected" if db else "not initialized",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check with database status."""
    db_status = "error"
    if db:
        try:
            # Test database connection
            with db.get_connection() as conn:
                conn.execute("SELECT 1")
            db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)}"

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        database=db_status,
        timestamp=datetime.now().isoformat(),
    )


# ===========================================
# Post Endpoints
# ===========================================

@app.post("/api/posts/generate", response_model=PostBatch)
async def generate_posts(request: GeneratePostsRequest):
    """
    Generate social media posts using LLM.

    This endpoint fetches context from Notion and generates posts via OpenRouter.
    """
    try:
        config = Config.from_env()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

    # Import here to avoid circular imports
    from notion_client import NotionClient, format_documents_as_context
    from llm_client import LLMClient
    from prompts import SYSTEM_PROMPT, build_user_prompt

    try:
        # Fetch context from Notion
        notion = NotionClient(token=config.notion_token)
        documents = notion.fetch_all_documents(config.notion_page_id)
        context = format_documents_as_context(documents)

        # Generate posts with LLM
        llm = LLMClient(api_key=config.openrouter_api_key)
        user_prompt = build_user_prompt(
            company_context=context,
            topic=request.topic,
            num_posts=request.num_posts,
        )
        batch = llm.generate_posts(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Store posts in database
        if db:
            for post in batch.posts:
                db.create_post(
                    content=post.text,
                    topic=batch.topic,
                    tone=post.tone.value,
                )

        return batch

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/api/posts", response_model=list[PostResponse])
async def list_posts(limit: int = Query(default=10, ge=1, le=100)):
    """Get recent posts from database."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    posts = db.get_recent_posts(limit=limit)
    return [
        PostResponse(
            id=p.id,
            content=p.content,
            topic=p.topic,
            tone=p.tone,
            character_count=p.character_count,
            created_at=str(p.created_at),
        )
        for p in posts
    ]


@app.get("/api/posts/{post_id}", response_model=PostResponse)
async def get_post(post_id: int):
    """Get a specific post by ID."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    post = db.get_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    return PostResponse(
        id=post.id,
        content=post.content,
        topic=post.topic,
        tone=post.tone,
        character_count=post.character_count,
        created_at=str(post.created_at),
    )


# ===========================================
# Approval Endpoints
# ===========================================

@app.post("/api/posts/{post_id}/approve", response_model=ApprovalResponse)
async def approve_post(post_id: int, request: ApprovalRequest):
    """Record an approval decision for a post."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    # Verify post exists
    post = db.get_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    # Record approval
    approval_id = db.record_approval(
        post_id=post_id,
        decision=request.decision,
        edited_content=request.edited_content,
        rejection_reason=request.rejection_reason,
    )

    return ApprovalResponse(
        id=approval_id,
        post_id=post_id,
        decision=request.decision,
        approved_at=datetime.now().isoformat(),
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get approval and post statistics."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    approval_stats = db.get_approval_stats()
    recent_posts = db.get_recent_posts(limit=100)

    return StatsResponse(
        total_posts=len(recent_posts),
        approval_stats=approval_stats,
        recent_posts_count=len(recent_posts),
    )


# ===========================================
# Mastodon Endpoints
# ===========================================

@app.post("/api/mastodon/post", response_model=MastodonPostResponse)
async def post_to_mastodon(request: MastodonPostRequest):
    """Post content to Mastodon."""
    try:
        config = Config.from_env()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

    if not config.mastodon_token:
        raise HTTPException(status_code=400, detail="Mastodon token not configured")

    from mastodon_client import MastodonClient

    mastodon = MastodonClient(
        token=config.mastodon_token,
        instance=config.mastodon_instance,
    )

    if not mastodon.verify_credentials():
        raise HTTPException(status_code=401, detail="Invalid Mastodon credentials")

    result = mastodon.post_status(request.text)

    # Record in database if post_id provided
    if db and request.post_id and result.success:
        db.record_published_post(
            post_id=request.post_id,
            mastodon_status_id=result.status_id,
            mastodon_url=result.url,
        )

    return MastodonPostResponse(
        success=result.success,
        url=result.url,
        error=result.error,
    )


# ===========================================
# RAG Endpoints
# ===========================================

@app.post("/api/rag/embed", response_model=RAGEmbedResponse)
async def embed_documents():
    """
    Fetch Notion documents and embed them into RAG database.

    This is a one-time operation (or run when documents change).
    """
    try:
        config = Config.from_env()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

    from notion_client import NotionClient
    from rag_client import RAGClient

    try:
        # Fetch documents from Notion
        notion = NotionClient(token=config.notion_token)
        documents = notion.fetch_all_documents(config.notion_page_id)

        # Embed into RAG database
        rag = RAGClient()
        total_chunks = rag.embed_notion_documents(documents)

        return RAGEmbedResponse(
            success=True,
            chunks_embedded=total_chunks,
            documents_processed=len(documents),
            message=f"Embedded {total_chunks} chunks from {len(documents)} documents",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


@app.post("/api/rag/search", response_model=RAGSearchResponse)
async def search_rag(request: RAGSearchRequest):
    """
    Search RAG database for relevant content.

    Uses hybrid search (BM25 + semantic similarity).
    """
    from rag_client import RAGClient

    try:
        rag = RAGClient()
        results = rag.hybrid_search(request.query, top_k=request.top_k)

        return RAGSearchResponse(
            query=request.query,
            results=[
                RAGSearchResult(
                    content=r.content,
                    source_type=r.source_type,
                    source_id=r.source_id,
                    score=r.final_score,
                )
                for r in results
            ],
            total_results=len(results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/api/posts/generate-rag", response_model=PostBatch)
async def generate_posts_with_rag(request: GeneratePostsRequest):
    """
    Generate social media posts using RAG for context retrieval.

    This endpoint uses only relevant context instead of all documents.
    """
    try:
        config = Config.from_env()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

    from rag_client import RAGClient
    from llm_client import LLMClient
    from prompts import SYSTEM_PROMPT, build_user_prompt

    try:
        # Get relevant context via RAG
        rag = RAGClient()
        context = rag.search(request.topic, top_k=5)

        if not context:
            raise HTTPException(
                status_code=400,
                detail="No relevant context found. Run /api/rag/embed first.",
            )

        # Generate posts with LLM
        llm = LLMClient(api_key=config.openrouter_api_key)
        user_prompt = build_user_prompt(
            company_context=context,
            topic=request.topic,
            num_posts=request.num_posts,
        )
        batch = llm.generate_posts(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Store posts in database and optionally embed for duplicate detection
        if db:
            for post in batch.posts:
                post_id = db.create_post(
                    content=post.text,
                    topic=batch.topic,
                    tone=post.tone.value,
                )

                # Embed post into RAG if requested
                if request.embed_posts:
                    rag.embed_generated_post(
                        post_text=post.text,
                        post_id=post_id,
                        topic=batch.topic,
                        tone=post.tone.value,
                    )

        return batch

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


class DuplicateCheckRequest(BaseModel):
    """Request to check for duplicate posts."""
    text: str = Field(..., description="Post text to check")
    threshold: float = Field(default=0.8, ge=0.5, le=1.0, description="Similarity threshold (0.5-1.0)")


class SimilarPost(BaseModel):
    """A similar post found in the database."""
    content: str
    similarity: float
    topic: Optional[str] = None


class DuplicateCheckResponse(BaseModel):
    """Response from duplicate check."""
    is_duplicate: bool
    similar_posts: list[SimilarPost]


@app.post("/api/posts/check-duplicate", response_model=DuplicateCheckResponse)
async def check_duplicate(request: DuplicateCheckRequest):
    """
    Check if a post is similar to previously generated posts.

    Useful for avoiding duplicate content.
    """
    from rag_client import RAGClient

    try:
        rag = RAGClient()
        similar = rag.find_similar_posts(
            text=request.text,
            threshold=request.threshold,
            limit=5,
        )

        return DuplicateCheckResponse(
            is_duplicate=len(similar) > 0,
            similar_posts=[
                SimilarPost(
                    content=s["content"],
                    similarity=s["similarity"],
                    topic=s["metadata"].get("topic"),
                )
                for s in similar
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate check error: {str(e)}")


# ===========================================
# Image Generation Endpoints
# ===========================================

@app.post("/api/images/generate", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """
    Generate a mascot image using Replicate.

    Can use a custom prompt or generate based on post text.
    """
    try:
        config = Config.from_env()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

    if not config.replicate_api_token:
        raise HTTPException(status_code=400, detail="REPLICATE_API_TOKEN not configured")

    from replicate_client import ReplicateClient

    replicate = ReplicateClient(api_token=config.replicate_api_token)

    if request.prompt:
        # Use custom prompt
        result = replicate.generate_image(
            prompt=request.prompt,
            aspect_ratio=request.aspect_ratio,
        )
    elif request.post_text:
        # Generate mascot based on post text
        result = replicate.generate_mascot_image(request.post_text)
    else:
        # Default mascot
        result = replicate.generate_mascot_image("friendly community")

    return ImageGenerationResponse(
        success=result.success,
        image_url=result.image_url,
        error=result.error,
    )


@app.post("/api/posts/generate-with-images", response_model=PostBatchWithImages)
async def generate_posts_with_images(request: GeneratePostsRequest):
    """
    Generate social media posts with mascot images.

    Uses RAG for context retrieval and generates an image for each post.
    """
    try:
        config = Config.from_env()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

    from rag_client import RAGClient
    from llm_client import LLMClient
    from replicate_client import ReplicateClient
    from prompts import SYSTEM_PROMPT, build_user_prompt

    # Check for Replicate token
    if not config.replicate_api_token:
        raise HTTPException(status_code=400, detail="REPLICATE_API_TOKEN not configured")

    try:
        # Get relevant context via RAG
        rag = RAGClient()
        context = rag.search(request.topic, top_k=5)

        if not context:
            raise HTTPException(
                status_code=400,
                detail="No relevant context found. Run /api/rag/embed first.",
            )

        # Generate posts with LLM
        llm = LLMClient(api_key=config.openrouter_api_key)
        user_prompt = build_user_prompt(
            company_context=context,
            topic=request.topic,
            num_posts=request.num_posts,
        )
        batch = llm.generate_posts(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Generate images for each post
        replicate = ReplicateClient(api_token=config.replicate_api_token)
        posts_with_images = []

        for post in batch.posts:
            # Generate mascot image
            img_result = replicate.generate_mascot_image(post.text)

            posts_with_images.append(PostWithImageResponse(
                text=post.text,
                tone=post.tone.value,
                intended_use=post.intended_use.value,
                image_url=img_result.image_url if img_result.success else None,
            ))

            # Store in database
            if db:
                post_id = db.create_post(
                    content=post.text,
                    topic=batch.topic,
                    tone=post.tone.value,
                )

                # Record image if generated
                if img_result.success:
                    db.record_generated_image(
                        image_url=img_result.image_url,
                        prompt=f"Mascot for: {post.text[:100]}",
                        post_id=post_id,
                    )

                # Embed post if requested
                if request.embed_posts:
                    rag.embed_generated_post(
                        post_text=post.text,
                        post_id=post_id,
                        topic=batch.topic,
                        tone=post.tone.value,
                    )

        return PostBatchWithImages(
            topic=batch.topic,
            posts=posts_with_images,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


# ===========================================
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
# ===========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
