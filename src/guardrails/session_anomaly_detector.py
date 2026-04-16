"""
Bonus Safety Layer 3: Session Anomaly Detector
Detects users sending many injection-like messages in one session
"""
import time
from collections import defaultdict, deque
from typing import Dict, Any, List
from google.adk.plugins import base_plugin
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types


class SessionAnomalyDetectorPlugin(base_plugin.BasePlugin):
    """Plugin that detects anomalous user behavior in sessions.

    This layer tracks user behavior patterns and flags users who send
    many injection-like messages in a short time. It's needed because
    individual injection detection might miss coordinated attacks, and
    this catches suspicious user behavior patterns.
    """

    def __init__(self, max_injection_per_session: int = 3, session_timeout: int = 3600):
        super().__init__(name="session_anomaly_detector")
        self.max_injection_per_session = max_injection_per_session
        self.session_timeout = session_timeout  # Session timeout in seconds

        # Track user sessions: user_id -> {"injections": deque, "start_time": timestamp}
        self.user_sessions = defaultdict(lambda: {
            "injections": deque(maxlen=20),  # Keep last 20 messages
            "start_time": time.time()
        })

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

    def _is_injection_like(self, text: str) -> bool:
        """Check if message looks like an injection attempt."""
        injection_keywords = [
            "ignore", "bypass", "override", "system prompt",
            "you are now", "act as", "pretend", "roleplay",
            "jailbreak", "dan mode", "uncensored",
            "bỏ qua", "vượt qua", "hệ thống", "bạn là"
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in injection_keywords)

    def _cleanup_old_sessions(self):
        """Remove expired sessions to prevent memory leaks."""
        current_time = time.time()
        expired_users = []

        for user_id, session_data in self.user_sessions.items():
            if current_time - session_data["start_time"] > self.session_timeout:
                expired_users.append(user_id)

        for user_id in expired_users:
            del self.user_sessions[user_id]

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Check for anomalous session behavior."""
        self.total_count += 1
        user_id = invocation_context.user_id if invocation_context else "anonymous"
        text = self._extract_text(user_message)

        # Cleanup old sessions periodically
        if self.total_count % 100 == 0:
            self._cleanup_old_sessions()

        session = self.user_sessions[user_id]

        # Check if this message looks like injection
        is_injection = self._is_injection_like(text)
        session["injections"].append({
            "text": text,
            "is_injection": is_injection,
            "timestamp": time.time()
        })

        # Count recent injections (last 10 messages)
        recent_injections = sum(1 for msg in list(session["injections"])[-10:]
                               if msg["is_injection"])

        if recent_injections >= self.max_injection_per_session:
            self.blocked_count += 1
            return self._create_response(
                f"🚫 Suspicious session activity detected. Too many injection-like messages in this session. Access temporarily blocked."
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "total_checked": self.total_count,
            "sessions_blocked": self.blocked_count,
            "active_sessions": len(self.user_sessions),
            "block_rate": self.blocked_count / max(1, self.total_count),
        }