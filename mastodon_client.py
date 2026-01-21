"""
Mastodon API client for posting statuses and searching.
"""

import httpx
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MastodonPost:
    """A Mastodon post from search results."""
    id: str
    content: str  # HTML stripped to plain text
    author: str
    author_handle: str
    url: str
    created_at: str


@dataclass
class PostResult:
    """Result of a Mastodon post attempt."""
    success: bool
    post_id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None


class MastodonClient:
    """Client for posting to Mastodon."""

    def __init__(self, token: str, instance: str = "https://mastodon.social"):
        """
        Initialize Mastodon client.

        Args:
            token: Mastodon access token (from app settings)
            instance: Mastodon instance URL (e.g., https://mastodon.social)
        """
        self.token = token
        self.instance = instance.rstrip("/")
        self.api_base = f"{self.instance}/api/v1"

    def upload_media(self, image_url: str, description: str = "") -> Optional[str]:
        """
        Upload media from a URL to Mastodon.

        Args:
            image_url: URL of the image to upload
            description: Alt text for the image

        Returns:
            Media ID if successful, None otherwise
        """
        if not self.token:
            return None

        # First, download the image
        try:
            with httpx.Client(timeout=60.0) as client:
                img_response = client.get(image_url)
                if img_response.status_code != 200:
                    print(f"DEBUG: Failed to download image: HTTP {img_response.status_code}")
                    return None

                image_data = img_response.content
                content_type = img_response.headers.get("content-type", "image/webp")

                # Upload to Mastodon
                url = f"{self.api_base}/media"
                headers = {"Authorization": f"Bearer {self.token}"}

                # Determine file extension from content type
                ext = "webp"
                if "png" in content_type:
                    ext = "png"
                elif "jpeg" in content_type or "jpg" in content_type:
                    ext = "jpg"

                files = {
                    "file": (f"image.{ext}", image_data, content_type),
                }
                data = {}
                if description:
                    data["description"] = description

                response = client.post(url, headers=headers, files=files, data=data)

                if response.status_code in (200, 202):
                    media_data = response.json()
                    return media_data.get("id")
                else:
                    print(f"DEBUG: Media upload failed: HTTP {response.status_code}")
                    try:
                        print(f"DEBUG: {response.json()}")
                    except Exception:
                        pass
                    return None

        except Exception as e:
            print(f"DEBUG: Media upload error: {e}")
            return None

    def post_status(
        self,
        text: str,
        visibility: str = "public",
        sensitive: bool = False,
        spoiler_text: Optional[str] = None,
        media_ids: Optional[List[str]] = None,
    ) -> PostResult:
        """
        Post a status to Mastodon.

        Args:
            text: The status text (max 500 chars for most instances)
            visibility: One of 'public', 'unlisted', 'private', 'direct'
            sensitive: Mark as sensitive content
            spoiler_text: Content warning text (if any)
            media_ids: List of media IDs to attach

        Returns:
            PostResult with success status and post details
        """
        if not self.token:
            return PostResult(
                success=False,
                error="No Mastodon token configured. Set MASTODON_TOKEN in .env"
            )

        url = f"{self.api_base}/statuses"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        payload = {
            "status": text,
            "visibility": visibility,
        }

        if sensitive:
            payload["sensitive"] = True
        if spoiler_text:
            payload["spoiler_text"] = spoiler_text
        if media_ids:
            payload["media_ids"] = media_ids

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    return PostResult(
                        success=True,
                        post_id=data.get("id"),
                        url=data.get("url"),
                    )
                else:
                    error_msg = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = error_data["error"]
                    except Exception:
                        pass
                    return PostResult(success=False, error=error_msg)

        except httpx.TimeoutException:
            return PostResult(success=False, error="Request timed out")
        except httpx.RequestError as e:
            return PostResult(success=False, error=f"Request failed: {str(e)}")

    def verify_credentials(self) -> bool:
        """
        Verify that the token is valid.

        Returns:
            True if credentials are valid, False otherwise
        """
        if not self.token:
            return False

        url = f"{self.api_base}/accounts/verify_credentials"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=headers)
                return response.status_code == 200
        except Exception:
            return False

    def search_posts(self, keyword: str, limit: int = 5) -> List[MastodonPost]:
        """
        Search for recent public posts by hashtag.

        Uses the hashtag timeline API which returns all public posts
        with the given hashtag (not just posts you've interacted with).

        Args:
            keyword: The hashtag to search (with or without #)
            limit: Maximum number of posts to return (default 5)

        Returns:
            List of MastodonPost objects
        """
        # Strip # if provided
        hashtag = keyword.lstrip("#")

        url = f"{self.api_base}/timelines/tag/{hashtag}"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        params = {
            "limit": limit,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=headers, params=params)

                if response.status_code == 200:
                    data = response.json()
                    posts = []
                    for status in data:
                        # Strip HTML tags from content
                        content = self._strip_html(status.get("content", ""))
                        account = status.get("account", {})
                        posts.append(MastodonPost(
                            id=status.get("id", ""),
                            content=content,
                            author=account.get("display_name", "Unknown"),
                            author_handle=f"@{account.get('acct', 'unknown')}",
                            url=status.get("url", ""),
                            created_at=status.get("created_at", ""),
                        ))
                    return posts
                else:
                    print(f"DEBUG: Hashtag search failed with status {response.status_code}")
                    return []

        except Exception as e:
            print(f"DEBUG: Hashtag search error: {e}")
            return []

    def reply_to_post(
        self,
        text: str,
        reply_to_id: str,
        visibility: str = "public",
    ) -> PostResult:
        """
        Post a reply to an existing status.

        Args:
            text: The reply text
            reply_to_id: The ID of the post to reply to
            visibility: One of 'public', 'unlisted', 'private', 'direct'

        Returns:
            PostResult with success status and post details
        """
        if not self.token:
            return PostResult(
                success=False,
                error="No Mastodon token configured"
            )

        url = f"{self.api_base}/statuses"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        payload = {
            "status": text,
            "in_reply_to_id": reply_to_id,
            "visibility": visibility,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    return PostResult(
                        success=True,
                        post_id=data.get("id"),
                        url=data.get("url"),
                    )
                else:
                    error_msg = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = error_data["error"]
                    except Exception:
                        pass
                    return PostResult(success=False, error=error_msg)

        except httpx.TimeoutException:
            return PostResult(success=False, error="Request timed out")
        except httpx.RequestError as e:
            return PostResult(success=False, error=f"Request failed: {str(e)}")

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags from a string."""
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Decode common HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
