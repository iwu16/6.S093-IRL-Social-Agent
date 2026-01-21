"""
Replicate API client for generating images with fine-tuned FLUX model.
"""

import httpx
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageResult:
    """Result of an image generation attempt."""
    success: bool
    image_url: Optional[str] = None
    error: Optional[str] = None


class ReplicateClient:
    """Client for generating images via Replicate API."""

    BASE_URL = "https://api.replicate.com/v1"
    MODEL_VERSION = "sundai-club/sidequest:ecbe4c24c735dcd1e8daafed0461ae7bf217c45c884e616dffa92fa42c510036"

    def __init__(self, api_token: str):
        """
        Initialize Replicate client.

        Args:
            api_token: Replicate API token (from replicate.com/account)
        """
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.0,
        output_format: str = "webp",
    ) -> ImageResult:
        """
        Generate an image using the fine-tuned FLUX model.

        Args:
            prompt: Text description for image generation
            aspect_ratio: Image aspect ratio (1:1, 16:9, 4:3, etc.)
            num_inference_steps: Number of diffusion steps (1-50, default 28)
            guidance_scale: How closely to follow prompt (0-10, default 3)
            output_format: Output format (webp, jpg, png)

        Returns:
            ImageResult with success status and image URL
        """
        if not self.api_token:
            return ImageResult(
                success=False,
                error="No Replicate API token configured. Set REPLICATE_API_TOKEN in .env"
            )

        # Create prediction
        url = f"{self.BASE_URL}/predictions"
        payload = {
            "version": self.MODEL_VERSION.split(":")[1],
            "input": {
                "prompt": prompt,
                "model": "dev",
                "aspect_ratio": aspect_ratio,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "output_format": output_format,
                "num_outputs": 1,
            }
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=self.headers, json=payload)

                if response.status_code != 201:
                    error_msg = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        if "detail" in error_data:
                            error_msg = error_data["detail"]
                    except Exception:
                        pass
                    return ImageResult(success=False, error=error_msg)

                prediction = response.json()
                prediction_id = prediction.get("id")

                # Poll for completion
                return self._wait_for_prediction(client, prediction_id)

        except httpx.TimeoutException:
            return ImageResult(success=False, error="Request timed out")
        except httpx.RequestError as e:
            return ImageResult(success=False, error=f"Request failed: {str(e)}")

    def _wait_for_prediction(
        self,
        client: httpx.Client,
        prediction_id: str,
        max_attempts: int = 60,
        poll_interval: float = 2.0,
    ) -> ImageResult:
        """
        Poll for prediction completion.

        Args:
            client: HTTP client
            prediction_id: The prediction ID to poll
            max_attempts: Maximum polling attempts
            poll_interval: Seconds between polls

        Returns:
            ImageResult with success status and image URL
        """
        url = f"{self.BASE_URL}/predictions/{prediction_id}"

        for _ in range(max_attempts):
            response = client.get(url, headers=self.headers)

            if response.status_code != 200:
                return ImageResult(
                    success=False,
                    error=f"Failed to check prediction status: HTTP {response.status_code}"
                )

            prediction = response.json()
            status = prediction.get("status")

            if status == "succeeded":
                output = prediction.get("output")
                if output and len(output) > 0:
                    return ImageResult(success=True, image_url=output[0])
                return ImageResult(success=False, error="No image in output")

            elif status == "failed":
                error = prediction.get("error", "Unknown error")
                return ImageResult(success=False, error=error)

            elif status == "canceled":
                return ImageResult(success=False, error="Prediction was canceled")

            # Still processing, wait and retry
            time.sleep(poll_interval)

        return ImageResult(success=False, error="Prediction timed out")

    def generate_mascot_image(self, post_text: str, mascot_name: str = "raccoon") -> ImageResult:
        """
        Generate a mascot image that matches the post content.

        Args:
            post_text: The social media post text (used for context)
            mascot_name: The mascot trigger word (default: raccoon)

        Returns:
            ImageResult with success status and image URL
        """
        # Build a prompt that incorporates the post theme with the mascot
        prompt = f"A cute {mascot_name} mascot, friendly and approachable, " \
                 f"representing connection and community, warm colors, " \
                 f"simple clean illustration style"

        return self.generate_image(prompt)
