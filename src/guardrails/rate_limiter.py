"""
Rate Limiter Plugin for Defense-in-Depth Pipeline
"""
import time
from collections import defaultdict, deque
from google.adk.plugins import base_plugin
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Plugin that limits requests per user using sliding window algorithm."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows = defaultdict(deque)  # user_id -> deque of timestamps
        self.blocked_count = 0
        self.total_count = 0

    def _extract_text(self, content: types.Content) -> str:
        """Extract plain text from a Content object."""
        text = ""
        if content and content.parts:
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    def _block_response(self, message: str) -> types.Content:
        """Create a Content object with a block message."""
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    def _calculate_wait_time(self, user_id: str) -> float:
        """Calculate how long user must wait before next request."""
        now = time.time()
        window = self.user_windows[user_id]

        # Remove expired timestamps
        while window and now - window[0] > self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            # Calculate wait time until oldest request expires
            oldest_timestamp = window[0]
            wait_time = self.window_seconds - (now - oldest_timestamp)
            return max(0, wait_time)

        return 0

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Check rate limit before processing user message.

        Returns:
            None if under limit (let it through),
            types.Content if rate limited (return block message)
        """
        self.total_count += 1
        user_id = invocation_context.user_id if invocation_context else "anonymous"
        now = time.time()

        window = self.user_windows[user_id]

        # Remove expired timestamps
        while window and now - window[0] > self.window_seconds:
            window.popleft()

        # Check if user has exceeded limit
        if len(window) >= self.max_requests:
            self.blocked_count += 1
            wait_time = self._calculate_wait_time(user_id)
            return self._block_response(
                f"🚫 Rate limit exceeded. Please wait {wait_time:.1f} seconds before trying again."
            )

        # Add current timestamp and allow request
        window.append(now)
        return None