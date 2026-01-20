"""
Notion API client for fetching company documents.
Supports reading from a parent page with subpages.
"""

import httpx
from dataclasses import dataclass


@dataclass
class NotionPage:
    """A simplified representation of a Notion page."""
    id: str
    title: str
    content: str


class NotionClient:
    """Client for reading company documents from Notion."""

    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"

    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": self.NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def get_child_pages(self, parent_page_id: str) -> list[str]:
        """
        Fetch all child page IDs from a parent page.

        Args:
            parent_page_id: The ID of the parent Notion page

        Returns:
            List of child page IDs
        """
        url = f"{self.BASE_URL}/blocks/{parent_page_id}/children"

        with httpx.Client() as client:
            response = client.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

        # Find all child_page blocks
        child_page_ids = []
        for block in data.get("results", []):
            if block.get("type") == "child_page":
                child_page_ids.append(block["id"])

        return child_page_ids

    def get_page_content(self, page_id: str) -> NotionPage:
        """
        Fetch a page's title and content blocks.

        Args:
            page_id: The Notion page ID

        Returns:
            NotionPage with extracted text content
        """
        # Get page metadata (for title)
        page_url = f"{self.BASE_URL}/pages/{page_id}"
        blocks_url = f"{self.BASE_URL}/blocks/{page_id}/children"

        with httpx.Client() as client:
            # Fetch page metadata
            page_response = client.get(page_url, headers=self.headers)
            page_response.raise_for_status()
            page_data = page_response.json()

            # Fetch content blocks
            blocks_response = client.get(blocks_url, headers=self.headers)
            blocks_response.raise_for_status()
            blocks_data = blocks_response.json()

        # Extract title from page properties
        title = self._extract_title(page_data)

        # Extract text from blocks
        content = self._extract_block_text(blocks_data.get("results", []))

        return NotionPage(id=page_id, title=title, content=content)

    def _extract_title(self, page_data: dict) -> str:
        """Extract page title from Notion page data."""
        properties = page_data.get("properties", {})

        # Try common title property names
        for key in ["title", "Title", "Name", "name"]:
            if key in properties:
                title_prop = properties[key]
                if title_prop.get("type") == "title":
                    title_array = title_prop.get("title", [])
                    if title_array:
                        return title_array[0].get("plain_text", "Untitled")

        return "Untitled"

    def _extract_block_text(self, blocks: list[dict]) -> str:
        """Extract plain text from Notion blocks."""
        texts = []

        for block in blocks:
            block_type = block.get("type")

            # Handle common block types
            if block_type in ["paragraph", "heading_1", "heading_2", "heading_3",
                              "bulleted_list_item", "numbered_list_item", "quote"]:
                rich_text = block.get(block_type, {}).get("rich_text", [])
                text = "".join(rt.get("plain_text", "") for rt in rich_text)
                if text:
                    texts.append(text)

        return "\n\n".join(texts)

    def fetch_all_documents(self, parent_page_id: str) -> list[NotionPage]:
        """
        Fetch all company documents from subpages of a parent page.

        Args:
            parent_page_id: The Notion parent page ID

        Returns:
            List of NotionPage objects with content
        """
        child_page_ids = self.get_child_pages(parent_page_id)
        documents = []

        for page_id in child_page_ids:
            try:
                page = self.get_page_content(page_id)
                documents.append(page)
            except Exception as e:
                print(f"Warning: Failed to fetch page {page_id}: {e}")

        return documents


def format_documents_as_context(documents: list[NotionPage]) -> str:
    """
    Format Notion documents into a single context string for the LLM.

    Args:
        documents: List of NotionPage objects

    Returns:
        Formatted string with all document content
    """
    sections = []

    for doc in documents:
        section = f"## {doc.title}\n\n{doc.content}"
        sections.append(section)

    return "\n\n---\n\n".join(sections)
