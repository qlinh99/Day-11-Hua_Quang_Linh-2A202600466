"""
Bonus Safety Layer 5: Hallucination Detector with Knowledge Base Cross-check
"""
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

from typing import Dict, Any, List, Optional
from google.adk.plugins import base_plugin
from google.genai import types


class HallucinationDetectorPlugin(base_plugin.BasePlugin):
    """Plugin that detects hallucinations by cross-checking against knowledge base.

    This layer verifies AI responses against a known knowledge base of
    banking facts. If the response contains claims that contradict or
    aren't supported by the knowledge base, it's flagged as potentially
    hallucinated. It's needed because LLMs can generate convincing but
    incorrect information.
    """

    def __init__(self, knowledge_base_path: Optional[str] = None, confidence_threshold: float = 0.6):
        super().__init__(name="hallucination_detector")
        self.confidence_threshold = confidence_threshold
        self.blocked_count = 0
        self.flagged_count = 0
        self.total_count = 0

        # Initialize knowledge base
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)

        if EMBEDDING_AVAILABLE:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self._initialize_embeddings()
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.model = None
        else:
            self.model = None

    def _load_knowledge_base(self, path: Optional[str] = None) -> List[str]:
        """Load banking knowledge base facts."""
        # Default knowledge base - banking facts that should be accurate
        default_kb = [
            "Savings interest rates vary between 3.5% to 6.5% per annum",
            "ATM withdrawal limit is typically 50 million VND per day",
            "Credit card applications require minimum income of 5 million VND per month",
            "Bank accounts require valid ID (CMND/CCCD) for opening",
            "Online transfers are processed within 1-2 business days",
            "Overdraft fees are 0.05% per day on exceeded amount",
            "Mortgage rates start from 6.5% per annum",
            "Bank operates from 8:00 AM to 5:00 PM Monday to Friday",
            "Minimum balance for savings account is 100,000 VND",
            "Loan approval takes 3-7 business days",
            "ATM fees are 2,000 VND per transaction for non-account holders",
            "Credit score ranges from 300 to 900 points",
            "Bank holidays include Tet, National Day, and Liberation Day",
            "Foreign currency exchange rates fluctuate daily",
            "Mobile banking app is available 24/7",
            "Account statements are sent monthly by email",
            "PIN must be 6 digits long",
            "Joint accounts require all holders' signatures",
            "Fixed deposits have higher interest rates than savings",
            "Bank has over 100 branches nationwide",
        ]

        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    custom_kb = [line.strip() for line in f if line.strip()]
                return default_kb + custom_kb
            except FileNotFoundError:
                print(f"Knowledge base file not found: {path}. Using default KB.")

        return default_kb

    def _initialize_embeddings(self):
        """Initialize embeddings for knowledge base."""
        if self.model and self.knowledge_base:
            self.kb_embeddings = self.model.encode(self.knowledge_base)

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    def _create_response(self, text: str) -> types.Content:
        """Create a Content object from text."""
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=text)],
        )

    def _check_factual_consistency_fallback(self, response_text: str) -> Dict[str, Any]:
        """Fallback consistency check using keyword matching."""
        # Simple keyword-based consistency check
        response_lower = response_text.lower()

        # Banking-specific keywords that indicate factual content
        banking_keywords = [
            "interest rate", "account", "transfer", "balance", "loan", "credit",
            "deposit", "withdraw", "atm", "card", "fee", "branch", "online",
            "mobile", "banking", "transaction", "payment", "currency", "exchange"
        ]

        # Potentially problematic keywords (could indicate hallucinations)
        risky_keywords = [
            "guarantee", "promise", "definitely", "always", "never", "impossible",
            "exact", "precise", "specific", "particular", "certain"
        ]

        # Count banking keywords
        banking_matches = sum(1 for keyword in banking_keywords if keyword in response_lower)
        risky_matches = sum(1 for keyword in risky_keywords if keyword in response_lower)

        # Simple heuristic: if response has banking keywords but also risky absolutes, flag it
        has_banking_content = banking_matches >= 2
        has_risky_language = risky_matches >= 1

        if has_banking_content and has_risky_language:
            confidence = 0.4  # Low confidence due to risky language
            safe = False
            reason = "Contains banking content with potentially risky absolute statements"
        elif has_banking_content:
            confidence = 0.8  # Good confidence for banking content
            safe = True
            reason = "Contains banking-related content"
        else:
            confidence = 0.6  # Neutral confidence
            safe = True
            reason = "No specific banking content detected"

        return {
            "safe": safe,
            "confidence": confidence,
            "inconsistent_claims": [] if safe else [{"sentence": "Response contains absolute statements", "confidence": confidence}],
            "reason": reason
        }

    def _check_factual_consistency(self, response_text: str) -> Dict[str, Any]:
        """Check if response is consistent with knowledge base."""
        if not self.model or not hasattr(self, 'kb_embeddings'):
            return self._check_factual_consistency_fallback(response_text)

        try:
            # Split response into sentences for granular checking
            sentences = [s.strip() for s in response_text.split('.') if s.strip()]

            total_confidence = 0
            checked_sentences = 0
            inconsistent_claims = []

            for sentence in sentences:
                if len(sentence) < 10:  # Skip very short sentences
                    continue

                sentence_embedding = self.model.encode([sentence])
                similarities = cosine_similarity(sentence_embedding, self.kb_embeddings)

                max_similarity = np.max(similarities)
                confidence = max_similarity

                # If confidence is low, this might be a hallucination
                if confidence < self.confidence_threshold:
                    inconsistent_claims.append({
                        "sentence": sentence,
                        "confidence": confidence
                    })

                total_confidence += confidence
                checked_sentences += 1

            avg_confidence = total_confidence / max(1, checked_sentences)

            return {
                "safe": len(inconsistent_claims) == 0,
                "confidence": avg_confidence,
                "inconsistent_claims": inconsistent_claims,
                "reason": f"Found {len(inconsistent_claims)} potentially inconsistent claims"
            }

        except Exception as e:
            print(f"Hallucination check error: {e}")
            return self._check_factual_consistency_fallback(response_text)

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response for potential hallucinations."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        consistency_check = self._check_factual_consistency(response_text)

        if not consistency_check["safe"]:
            self.flagged_count += 1

            # For now, we'll just log but not block - in production you might block or flag
            # For the bonus, we'll add a disclaimer
            disclaimer = "\n\n⚠️ **Disclaimer:** Some information in this response may not be fully verified against our knowledge base."

            # Modify the response to add disclaimer
            if hasattr(llm_response, "content") and llm_response.content:
                for part in llm_response.content.parts:
                    if hasattr(part, "text") and part.text:
                        part.text += disclaimer
                        break

        return llm_response

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "total_checked": self.total_count,
            "hallucinations_flagged": self.flagged_count,
            "hallucination_rate": self.flagged_count / max(1, self.total_count),
            "threshold": self.confidence_threshold,
            "kb_size": len(self.knowledge_base),
            "model_available": self.model is not None,
        }