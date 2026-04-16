"""
Bonus Safety Layer 1: Toxicity Classifier using Perspective API
"""
import aiohttp
import json
from typing import Dict, Any, Optional
from google.adk.plugins import base_plugin
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types


class ToxicityClassifierPlugin(base_plugin.BasePlugin):
    """Plugin that detects toxic content using Google Perspective API.

    This layer catches harmful, abusive, or inappropriate content that
    other layers might miss. It's needed because input guardrails focus
    on injection/topic filtering, but toxicity detection requires
    understanding sentiment and intent.
    """

    def __init__(self, api_key: str, threshold: float = 0.7):
        super().__init__(name="toxicity_classifier")
        self.api_key = api_key
        self.threshold = threshold  # Toxicity score threshold (0-1)
        self.api_url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
        self.blocked_count = 0
        self.total_count = 0

    def _extract_text(self, content: types.Content) -> str:
        """Extract plain text from Content object."""
        text = ""
        if content and content.parts:
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    def _create_response(self, text: str) -> types.Content:
        """Create a Content object from text."""
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=text)],
        )

    async def _analyze_toxicity(self, text: str) -> Optional[float]:
        """Analyze text toxicity using Perspective API."""
        if not text.strip():
            return 0.0

        payload = {
            "comment": {"text": text},
            "languages": ["en", "vi"],
            "requestedAttributes": {
                "TOXICITY": {},
                "SEVERE_TOXICITY": {},
                "IDENTITY_ATTACK": {},
                "INSULT": {},
                "THREAT": {}
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}?key={self.api_key}",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Return the highest toxicity score
                        scores = []
                        for attr in ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "THREAT"]:
                            if attr in data.get("attributeScores", {}):
                                score = data["attributeScores"][attr]["summaryScore"]["value"]
                                scores.append(score)
                        return max(scores) if scores else 0.0
                    else:
                        print(f"Perspective API error: {response.status}")
                        return None
        except Exception as e:
            print(f"Perspective API request failed: {e}")
            return None

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Check user message for toxicity before processing."""
        self.total_count += 1
        text = self._extract_text(user_message)

        toxicity_score = await self._analyze_toxicity(text)

        if toxicity_score is not None and toxicity_score >= self.threshold:
            self.blocked_count += 1
            return self._create_response(
                f"🚫 Toxic content detected (score: {toxicity_score:.2f}). Request blocked for maintaining a safe environment."
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "total_checked": self.total_count,
            "toxicity_blocked": self.blocked_count,
            "toxicity_rate": self.blocked_count / max(1, self.total_count),
        }