"""
Bonus Safety Layer 4: Embedding Similarity Filter for Topic Clustering
"""
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

from typing import Dict, Any, List
from google.adk.plugins import base_plugin
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types


class EmbeddingSimilarityPlugin(base_plugin.BasePlugin):
    """Plugin that filters queries based on semantic similarity to banking topics.

    This layer uses embeddings to measure how similar a query is to the
    expected banking/finance domain. Queries too far from the topic cluster
    are blocked. It's needed because keyword-based topic filters can be
    bypassed with synonyms or paraphrases.
    """

    def __init__(self, similarity_threshold: float = 0.3, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(name="embedding_similarity")
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name

        if EMBEDDING_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self._initialize_topic_embeddings()
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.model = None
        else:
            self.model = None
            self._initialize_fallback_topics()

        self.blocked_count = 0
        self.total_count = 0

    def _initialize_fallback_topics(self):
        """Initialize keyword-based topic matching for fallback."""
        self.fallback_topics = [
            # Banking keywords
            "bank", "account", "transfer", "money", "deposit", "withdraw", "balance",
            "loan", "credit", "interest", "rate", "saving", "checking", "atm",
            "card", "payment", "transaction", "fee", "branch", "online", "mobile",
            "app", "finance", "financial", "currency", "exchange", "wire", "swift",

            # Vietnamese banking terms
            "ngân hàng", "tài khoản", "chuyển tiền", "tiền", "gửi tiền", "rút tiền",
            "số dư", "khoản vay", "tín dụng", "lãi suất", "tiết kiệm", "kiểm tra",
            "atm", "thẻ", "thanh toán", "giao dịch", "phí", "chi nhánh", "trực tuyến",
            "di động", "ứng dụng", "tài chính", "tiền tệ", "đổi tiền", "chuyển khoản"
        ]

    def _initialize_topic_embeddings(self):
        """Initialize embeddings for banking/finance topics."""
        if not self.model:
            return

        # Core banking topics for embedding comparison
        banking_topics = [
            "What is the savings interest rate?",
            "How do I open a bank account?",
            "I want to transfer money to another account",
            "What are the ATM withdrawal limits?",
            "How do I apply for a credit card?",
            "What is my account balance?",
            "How do I pay my bills online?",
            "What are the loan interest rates?",
            "How do I deposit money?",
            "What is the minimum balance requirement?",
            "How do I check my transaction history?",
            "What are the bank fees?",
            "How do I change my PIN?",
            "What is the process for getting a mortgage?",
            "How do I set up online banking?",
            "What are the different types of savings accounts?",
            "How do I report a lost card?",
            "What is the bank's routing number?",
            "How do I apply for a personal loan?",
            "What are the requirements for opening a joint account?",
            "Fixed deposits have higher interest rates than savings",
            "Bank has over 100 branches nationwide",
        ]

        self.topic_embeddings = self.model.encode(banking_topics)

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

    def _calculate_similarity_fallback(self, query: str) -> float:
        """Fallback similarity calculation using keyword matching."""
        if not hasattr(self, 'fallback_topics'):
            return 1.0

        query_lower = query.lower()
        matching_words = 0
        total_words = len(query.split())

        if total_words == 0:
            return 0.0

        for topic in self.fallback_topics:
            if topic.lower() in query_lower:
                matching_words += 1

        # Return a similarity score based on keyword matches
        similarity = min(matching_words / max(1, total_words), 1.0)
        return similarity

    def _calculate_similarity(self, query: str) -> float:
        """Calculate maximum similarity to banking topics."""
        if not self.model or not hasattr(self, 'topic_embeddings'):
            return self._calculate_similarity_fallback(query)

        try:
            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.topic_embeddings)
            max_similarity = np.max(similarities)
            return max_similarity
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return self._calculate_similarity_fallback(query)

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Check semantic similarity to banking topics."""
        self.total_count += 1
        text = self._extract_text(user_message)

        similarity_score = self._calculate_similarity(text)

        if similarity_score < self.similarity_threshold:
            self.blocked_count += 1
            return self._create_response(
                f"🚫 Query appears off-topic (similarity: {similarity_score:.2f}). This AI assistant only handles banking and financial services."
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "total_checked": self.total_count,
            "similarity_blocked": self.blocked_count,
            "similarity_block_rate": self.blocked_count / max(1, self.total_count),
            "threshold": self.similarity_threshold,
            "model_available": self.model is not None,
        }