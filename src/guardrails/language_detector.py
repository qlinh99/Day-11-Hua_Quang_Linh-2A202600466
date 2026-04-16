"""
Bonus Safety Layer 2: Language Detection to Block Unsupported Languages
"""
try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("langdetect not available. Language detection will use fallback method.")

from typing import Dict, Any, Set
from google.adk.plugins import base_plugin
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types


class LanguageDetectionPlugin(base_plugin.BasePlugin):
    """Plugin that blocks queries in unsupported languages.

    This layer ensures the AI only responds to queries in supported languages
    (English and Vietnamese for banking context). It's needed because the
    banking AI is trained for specific languages and off-topic languages
    might bypass topic filters or cause poor responses.
    """

    def __init__(self, supported_languages: Set[str] = {"en", "vi"}):
        super().__init__(name="language_detector")
        self.supported_languages = supported_languages
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

    def _detect_language_fallback(self, text: str) -> str:
        """Fallback language detection using simple heuristics."""
        text = text.lower().strip()

        # Vietnamese indicators
        vi_indicators = ['và', 'của', 'là', 'được', 'có', 'cho', 'với', 'từ', 'để', 'hoặc',
                        'tôi', 'bạn', 'anh', 'chị', 'ông', 'bà', 'thế nào', 'tại sao',
                        'tiếng việt', 'việt nam', 'hà nội', 'sài gòn', 'đà nẵng']

        # Count Vietnamese words
        vi_count = sum(1 for word in vi_indicators if word in text)

        # Vietnamese-specific characters
        vi_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
        vi_char_count = sum(1 for char in text if char in vi_chars)

        if vi_count >= 2 or vi_char_count >= 3:
            return "vi"

        # Default to English for other cases
        return "en"

    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        if not text.strip():
            return "unknown"

        if LANGDETECT_AVAILABLE:
            try:
                # Clean text for better detection
                clean_text = text.strip()
                if len(clean_text) < 10:  # Too short for reliable detection
                    return self._detect_language_fallback(clean_text)

                detected_lang = detect(clean_text)
                return detected_lang
            except (LangDetectError, Exception):
                return self._detect_language_fallback(text)
        else:
            return self._detect_language_fallback(text)

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Check if user message is in a supported language."""
        self.total_count += 1
        text = self._extract_text(user_message)

        detected_lang = self._detect_language(text)

        if detected_lang not in self.supported_languages:
            self.blocked_count += 1
            supported_list = ", ".join(self.supported_languages)
            return self._create_response(
                f"🚫 Language '{detected_lang}' is not supported. This AI assistant only responds to queries in: {supported_list}."
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "total_checked": self.total_count,
            "language_blocked": self.blocked_count,
            "language_block_rate": self.blocked_count / max(1, self.total_count),
            "supported_languages": list(self.supported_languages),
            "langdetect_available": LANGDETECT_AVAILABLE,
        }