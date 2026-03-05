"""
Token Usage Tracker for PyChat - Tracks token usage and estimates costs per conversation.
Provides per-thread, per-provider, and session-level usage statistics.
"""

import sqlite3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (input/output) as of early 2026 - approximate
MODEL_PRICING = {
    # OpenAI
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3.5-haiku": {"input": 0.80, "output": 4.0},
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    # Gemini
    "gemini-pro": {"input": 0.50, "output": 1.50},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # Ollama (local, free)
    "_ollama_default": {"input": 0.0, "output": 0.0},
}

# Average characters per token by provider
CHARS_PER_TOKEN = {
    "openai": 4.0,
    "anthropic": 3.5,
    "gemini": 4.0,
    "ollama": 4.0,
}


class TokenTracker(QObject):
    """Tracks token usage and costs across conversations"""

    usage_updated = pyqtSignal(dict)  # Emitted when usage stats change

    def __init__(self, db_path: str = "chat_history.db"):
        super().__init__()
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        """Create the token tracking table if it doesn't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id INTEGER,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    estimated_cost_usd REAL DEFAULT 0.0,
                    message_role TEXT,
                    FOREIGN KEY (thread_id) REFERENCES threads(id)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_token_usage_thread
                ON token_usage(thread_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp
                ON token_usage(timestamp)
            """)
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Failed to create token_usage table: {e}")

    def estimate_tokens(self, text: str, provider: str) -> int:
        """Estimate token count from text length"""
        if not text:
            return 0
        chars_per_token = CHARS_PER_TOKEN.get(provider, 4.0)
        return max(1, int(len(text) / chars_per_token))

    def get_model_pricing(self, model: str, provider: str) -> Dict[str, float]:
        """Get pricing for a model, with fuzzy matching"""
        if provider == "ollama":
            return MODEL_PRICING["_ollama_default"]

        # Try exact match first
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]

        # Try prefix match (e.g., "gpt-4-turbo-2024-04-09" -> "gpt-4-turbo")
        for known_model, pricing in MODEL_PRICING.items():
            if known_model.startswith("_"):
                continue
            if model.startswith(known_model) or known_model in model:
                return pricing

        # Default: return zero (unknown model)
        return {"input": 0.0, "output": 0.0}

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model: str, provider: str
    ) -> float:
        """Calculate estimated cost in USD"""
        pricing = self.get_model_pricing(model, provider)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def record_usage(
        self,
        thread_id: int,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        message_role: str = "exchange",
    ):
        """Record token usage for a message exchange"""
        input_tokens = self.estimate_tokens(input_text, provider)
        output_tokens = self.estimate_tokens(output_text, provider)
        cost = self.calculate_cost(input_tokens, output_tokens, model, provider)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO token_usage
                (thread_id, timestamp, provider, model, input_tokens, output_tokens,
                 estimated_cost_usd, message_role)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    thread_id,
                    datetime.utcnow().isoformat(),
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    cost,
                    message_role,
                ),
            )
            conn.commit()
            conn.close()

            # Emit updated stats
            stats = self.get_thread_usage(thread_id)
            self.usage_updated.emit(stats)

        except sqlite3.Error as e:
            logger.error(f"Failed to record token usage: {e}")

    def get_thread_usage(self, thread_id: int) -> Dict[str, Any]:
        """Get usage statistics for a specific thread"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    COALESCE(SUM(input_tokens), 0) as total_input,
                    COALESCE(SUM(output_tokens), 0) as total_output,
                    COALESCE(SUM(estimated_cost_usd), 0) as total_cost,
                    COUNT(*) as message_count
                FROM token_usage WHERE thread_id = ?
            """,
                (thread_id,),
            )
            row = cursor.fetchone()
            conn.close()

            return {
                "thread_id": thread_id,
                "input_tokens": row[0],
                "output_tokens": row[1],
                "total_tokens": row[0] + row[1],
                "estimated_cost_usd": round(row[2], 4),
                "message_count": row[3],
            }
        except sqlite3.Error as e:
            logger.error(f"Failed to get thread usage: {e}")
            return {
                "thread_id": thread_id,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "message_count": 0,
            }

    def get_provider_usage(
        self, provider: Optional[str] = None, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get usage statistics grouped by provider"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

            if provider:
                cursor.execute(
                    """
                    SELECT provider, model,
                        SUM(input_tokens) as total_input,
                        SUM(output_tokens) as total_output,
                        SUM(estimated_cost_usd) as total_cost,
                        COUNT(*) as exchanges
                    FROM token_usage
                    WHERE provider = ? AND timestamp >= ?
                    GROUP BY provider, model
                    ORDER BY total_cost DESC
                """,
                    (provider, cutoff),
                )
            else:
                cursor.execute(
                    """
                    SELECT provider, model,
                        SUM(input_tokens) as total_input,
                        SUM(output_tokens) as total_output,
                        SUM(estimated_cost_usd) as total_cost,
                        COUNT(*) as exchanges
                    FROM token_usage
                    WHERE timestamp >= ?
                    GROUP BY provider, model
                    ORDER BY total_cost DESC
                """,
                    (cutoff,),
                )

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get provider usage: {e}")
            return []

    def get_session_summary(self) -> Dict[str, Any]:
        """Get overall session usage summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total usage
            cursor.execute("""
                SELECT
                    COALESCE(SUM(input_tokens), 0),
                    COALESCE(SUM(output_tokens), 0),
                    COALESCE(SUM(estimated_cost_usd), 0),
                    COUNT(DISTINCT thread_id),
                    COUNT(*)
                FROM token_usage
            """)
            row = cursor.fetchone()

            # Today's usage
            today = datetime.utcnow().strftime("%Y-%m-%d")
            cursor.execute(
                """
                SELECT
                    COALESCE(SUM(input_tokens), 0),
                    COALESCE(SUM(output_tokens), 0),
                    COALESCE(SUM(estimated_cost_usd), 0)
                FROM token_usage WHERE timestamp >= ?
            """,
                (today,),
            )
            today_row = cursor.fetchone()

            conn.close()

            return {
                "total_input_tokens": row[0],
                "total_output_tokens": row[1],
                "total_tokens": row[0] + row[1],
                "total_cost_usd": round(row[2], 4),
                "threads_used": row[3],
                "total_exchanges": row[4],
                "today_input_tokens": today_row[0],
                "today_output_tokens": today_row[1],
                "today_cost_usd": round(today_row[2], 4),
            }
        except sqlite3.Error as e:
            logger.error(f"Failed to get session summary: {e}")
            return {}

    def format_cost(self, cost_usd: float) -> str:
        """Format cost for display"""
        if cost_usd == 0:
            return "Free (local)"
        elif cost_usd < 0.01:
            return f"< $0.01"
        elif cost_usd < 1.0:
            return f"${cost_usd:.4f}"
        else:
            return f"${cost_usd:.2f}"

    def format_tokens(self, count: int) -> str:
        """Format token count for display"""
        if count < 1000:
            return str(count)
        elif count < 1_000_000:
            return f"{count/1000:.1f}K"
        else:
            return f"{count/1_000_000:.2f}M"
