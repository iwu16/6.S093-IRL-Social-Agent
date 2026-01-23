-- IRL Social Agent Database Schema
-- SQLite database for tracking posts, approvals, and analytics

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- ============================================
-- TABLES
-- ============================================

-- Table: posts
-- Stores all generated social media posts
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    topic TEXT,
    tone TEXT,
    source_doc_id TEXT,  -- Notion document ID that inspired this post
    character_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: approvals
-- Tracks HITL approval decisions
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('approve', 'reject', 'edit')),
    edited_content TEXT,  -- If decision is 'edit', stores the modified text
    rejection_reason TEXT,  -- If decision is 'reject', stores the reason
    approved_by TEXT,  -- Telegram user who approved
    approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
);

-- Table: published_posts
-- Records posts that were actually published to Mastodon
CREATE TABLE IF NOT EXISTS published_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL,
    approval_id INTEGER,
    mastodon_status_id TEXT,  -- ID returned by Mastodon API
    mastodon_url TEXT,  -- URL to the published post
    published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
    FOREIGN KEY (approval_id) REFERENCES approvals(id) ON DELETE SET NULL
);

-- Table: generated_images
-- Tracks mascot images generated via Replicate
CREATE TABLE IF NOT EXISTS generated_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER,
    prompt TEXT NOT NULL,
    image_url TEXT NOT NULL,
    model_version TEXT,
    generation_time_seconds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE SET NULL
);

-- Table: notion_documents
-- Cache of fetched Notion documents
CREATE TABLE IF NOT EXISTS notion_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    notion_page_id TEXT NOT NULL UNIQUE,
    title TEXT,
    content TEXT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: engagement_metrics
-- Track engagement on published posts (for future analytics)
CREATE TABLE IF NOT EXISTS engagement_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    published_post_id INTEGER NOT NULL,
    likes INTEGER DEFAULT 0,
    reblogs INTEGER DEFAULT 0,
    replies INTEGER DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (published_post_id) REFERENCES published_posts(id) ON DELETE CASCADE
);

-- ============================================
-- INDICES
-- ============================================

-- Index for faster post lookups by topic
CREATE INDEX IF NOT EXISTS idx_posts_topic ON posts(topic);

-- Index for faster post lookups by creation date
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);

-- Index for faster approval lookups by decision
CREATE INDEX IF NOT EXISTS idx_approvals_decision ON approvals(decision);

-- Index for faster approval lookups by post
CREATE INDEX IF NOT EXISTS idx_approvals_post_id ON approvals(post_id);

-- Index for published posts by Mastodon status ID
CREATE INDEX IF NOT EXISTS idx_published_mastodon_id ON published_posts(mastodon_status_id);

-- Index for Notion documents by page ID
CREATE INDEX IF NOT EXISTS idx_notion_page_id ON notion_documents(notion_page_id);

-- ============================================
-- VIEWS
-- ============================================

-- View: recent_posts_with_status
-- Shows recent posts with their approval and publication status
CREATE VIEW IF NOT EXISTS recent_posts_with_status AS
SELECT
    p.id AS post_id,
    p.content,
    p.topic,
    p.tone,
    p.created_at,
    a.decision AS approval_decision,
    a.approved_at,
    pp.mastodon_url,
    pp.published_at
FROM posts p
LEFT JOIN approvals a ON p.id = a.post_id
LEFT JOIN published_posts pp ON p.id = pp.post_id
ORDER BY p.created_at DESC;

-- View: approval_stats
-- Aggregated statistics on approval decisions
CREATE VIEW IF NOT EXISTS approval_stats AS
SELECT
    decision,
    COUNT(*) as count,
    DATE(approved_at) as date
FROM approvals
GROUP BY decision, DATE(approved_at)
ORDER BY date DESC;

-- ============================================
-- TRIGGERS
-- ============================================

-- Trigger: Update posts.updated_at on modification
CREATE TRIGGER IF NOT EXISTS update_posts_timestamp
AFTER UPDATE ON posts
BEGIN
    UPDATE posts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Trigger: Update notion_documents.updated_at on modification
CREATE TRIGGER IF NOT EXISTS update_notion_docs_timestamp
AFTER UPDATE ON notion_documents
BEGIN
    UPDATE notion_documents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================
-- RAG EMBEDDINGS TABLES
-- ============================================

-- Table: embeddings_meta
-- Stores document chunks with metadata (content for FTS5, linked to vectors by rowid)
CREATE TABLE IF NOT EXISTS embeddings_meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,  -- 'notion_doc', 'generated_post', etc.
    source_id TEXT,  -- Original document/post ID
    content TEXT NOT NULL,  -- The actual text chunk
    metadata TEXT,  -- JSON metadata (section_title, source_file, etc.)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Note: vec_embeddings table is created via sqlite-vec extension in Python
-- It stores 384-dimensional vectors from MiniLM-L6-v2
-- Schema: CREATE VIRTUAL TABLE vec_embeddings USING vec0(embedding float[384] distance_metric=cosine)

-- FTS5 virtual table for BM25 keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_fts USING fts5(
    content,
    source_type,
    source_id,
    content='embeddings_meta',
    content_rowid='id'
);

-- Trigger: Keep FTS5 in sync when inserting into embeddings_meta
CREATE TRIGGER IF NOT EXISTS embeddings_ai AFTER INSERT ON embeddings_meta BEGIN
    INSERT INTO embeddings_fts(rowid, content, source_type, source_id)
    VALUES (new.id, new.content, new.source_type, new.source_id);
END;

-- Trigger: Keep FTS5 in sync when deleting from embeddings_meta
CREATE TRIGGER IF NOT EXISTS embeddings_ad AFTER DELETE ON embeddings_meta BEGIN
    INSERT INTO embeddings_fts(embeddings_fts, rowid, content, source_type, source_id)
    VALUES ('delete', old.id, old.content, old.source_type, old.source_id);
END;

-- Index for embeddings by source
CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings_meta(source_type, source_id);
