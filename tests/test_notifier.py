"""
Unit tests for alerts/notifier.py

All tests use in-memory DataFrames and unittest.mock — no network or disk I/O.
"""

import pathlib
import sys
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from alerts.notifier import EmailNotifier, NotificationManager, TelegramNotifier

# ── Shared fixtures ────────────────────────────────────────────────────────

_CFG_BASE = {
    "alerts": {
        "email_enabled": False,
        "email_smtp_server": "smtp.gmail.com",
        "email_port": 587,
        "telegram_enabled": False,
        "min_confidence_to_alert": 0.65,
    }
}

_CARDS = pd.DataFrame([
    {
        "ticker":         "AAPL",
        "signal_name":    "LONG",
        "confidence":     0.75,
        "price":          150.00,
        "limit_entry":    149.50,
        "stop_loss":      145.00,
        "take_profit":    158.50,
        "rr_ratio":       2.0,
        "position_units": 10,
        "position_value": 1495.00,
        "position_pct":   2.99,
        "regime_name":    "bull",
        "adx":            25.0,
        "rsi":            55.0,
    },
    {
        "ticker":         "MSFT",
        "signal_name":    "SHORT",
        "confidence":     0.80,
        "price":          300.00,
        "limit_entry":    300.50,
        "stop_loss":      308.00,
        "take_profit":    285.50,
        "rr_ratio":       2.0,
        "position_units": 5,
        "position_value": 1502.50,
        "position_pct":   3.00,
        "regime_name":    "bear",
        "adx":            28.0,
        "rsi":            62.0,
    },
])


def _make_mgr(extra_alerts: dict | None = None) -> NotificationManager:
    """Build a NotificationManager with env-loading patched out."""
    cfg = {
        "alerts": {
            "email_enabled":            False,
            "telegram_enabled":         False,
            "min_confidence_to_alert":  0.65,
            **(extra_alerts or {}),
        }
    }
    with patch("alerts.notifier._load_env"):
        return NotificationManager(cfg)


# ── EmailNotifier ──────────────────────────────────────────────────────────


class TestEmailNotifier:

    def _make(self, user: str = "", password: str = "") -> EmailNotifier:
        """Construct an EmailNotifier with explicit credentials."""
        with patch("alerts.notifier._load_env"):
            n = EmailNotifier(_CFG_BASE)
        n.user     = user
        n.password = password
        return n

    def test_no_creds_returns_false(self):
        """send_signal_alert returns False when credentials are missing."""
        n = self._make(user="", password="")
        assert n.send_signal_alert(_CARDS) is False

    def test_smtp_success_returns_true(self):
        """send_signal_alert returns True when SMTP succeeds (mocked)."""
        n = self._make(user="test@example.com", password="secret")
        # MagicMock handles the context-manager protocol automatically
        with patch("smtplib.SMTP"):
            result = n.send_signal_alert(_CARDS)
        assert result is True

    def test_smtp_exception_returns_false(self):
        """send_signal_alert returns False when SMTP raises an exception."""
        n = self._make(user="test@example.com", password="secret")
        with patch("smtplib.SMTP", side_effect=OSError("connection refused")):
            result = n.send_signal_alert(_CARDS)
        assert result is False

    def test_build_html_contains_ticker(self):
        """_build_html returns an HTML string containing both ticker symbols."""
        n = self._make()
        html = n._build_html(_CARDS)
        assert isinstance(html, str)
        assert "AAPL" in html
        assert "MSFT" in html


# ── TelegramNotifier ──────────────────────────────────────────────────────


class TestTelegramNotifier:

    def _make(self, token: str = "", chat_id: str = "") -> TelegramNotifier:
        """Construct a TelegramNotifier with explicit credentials."""
        with patch("alerts.notifier._load_env"):
            n = TelegramNotifier(_CFG_BASE)
        n.token   = token
        n.chat_id = chat_id
        return n

    def test_no_creds_returns_false(self):
        """send_signal_alert returns False when token/chat_id are missing."""
        n = self._make(token="", chat_id="")
        assert n.send_signal_alert(_CARDS.iloc[0]) is False

    def test_success_returns_true(self):
        """send_signal_alert returns True on a mocked 200 response."""
        n = self._make(token="tok123", chat_id="chat456")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = n.send_signal_alert(_CARDS.iloc[0])
        assert result is True
        mock_post.assert_called_once()

    def test_exception_returns_false(self):
        """send_signal_alert returns False when requests.post raises."""
        n = self._make(token="tok123", chat_id="chat456")
        with patch("requests.post", side_effect=OSError("timeout")):
            result = n.send_signal_alert(_CARDS.iloc[0])
        assert result is False

    def test_format_message_contains_required_fields(self):
        """_format_message output contains ticker, confidence, entry, stop, target."""
        n = self._make()
        card = _CARDS.iloc[0]  # AAPL LONG
        msg  = n._format_message(card)
        assert "AAPL"   in msg            # ticker
        assert "75%"    in msg            # confidence 0.75 → "75%"
        assert "149.50" in msg            # limit_entry
        assert "145.00" in msg            # stop_loss
        assert "158.50" in msg            # take_profit


# ── NotificationManager ───────────────────────────────────────────────────


class TestNotificationManager:

    def test_empty_df_returns_zero_summary(self):
        """send_alerts on an empty DataFrame returns all-zero summary dict."""
        mgr    = _make_mgr()
        result = mgr.send_alerts(pd.DataFrame())
        assert result == {"sent": 0, "skipped_conf": 0, "skipped_dedup": 0}

    def test_filters_low_confidence(self):
        """Cards below min_conf increment skipped_conf; none are dispatched."""
        mgr          = _make_mgr()
        mgr.min_conf = 0.90  # both test cards (0.75, 0.80) fall below this
        with patch.object(mgr, "_load_sent_today", return_value=set()):
            result = mgr.send_alerts(_CARDS)
        assert result["skipped_conf"] == 2
        assert result["sent"]         == 0
        assert result["skipped_dedup"] == 0

    def test_deduplicates_already_sent(self):
        """Ticker+signal pair already sent today increments skipped_dedup."""
        mgr          = _make_mgr()
        mgr.min_conf = 0.70  # both cards (0.75, 0.80) pass confidence filter
        already_sent = {("AAPL", "LONG"), ("MSFT", "SHORT")}
        with patch.object(mgr, "_load_sent_today", return_value=already_sent):
            result = mgr.send_alerts(_CARDS)
        assert result["skipped_dedup"] == 2
        assert result["sent"]          == 0
        assert result["skipped_conf"]  == 0

    def test_calls_telegram_notifier_per_card(self):
        """Telegram notifier is invoked once for each qualifying card."""
        mgr = _make_mgr({"telegram_enabled": True, "min_confidence_to_alert": 0.70})
        mgr.telegram_on = True
        mock_tg = MagicMock()
        mock_tg.send_signal_alert.return_value = True
        mgr.telegram_notifier = mock_tg
        with patch.object(mgr, "_load_sent_today", return_value=set()):
            with patch.object(mgr, "_log_alert"):
                result = mgr.send_alerts(_CARDS)
        assert mock_tg.send_signal_alert.call_count == 2
        assert result["sent"] == 2

    def test_calls_email_notifier_once_for_batch(self):
        """Email notifier is called once with all qualifying cards as a batch."""
        mgr = _make_mgr({"email_enabled": True, "min_confidence_to_alert": 0.70})
        mgr.email_on = True
        mock_em = MagicMock()
        mock_em.send_signal_alert.return_value = True
        mgr.email_notifier = mock_em
        with patch.object(mgr, "_load_sent_today", return_value=set()):
            with patch.object(mgr, "_log_alert"):
                result = mgr.send_alerts(_CARDS)
        mock_em.send_signal_alert.assert_called_once()
        assert result["sent"] == 2

    def test_load_sent_today_absent(self):
        """_load_sent_today returns empty set when the log file does not exist."""
        mgr       = _make_mgr()
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with patch("alerts.notifier._LOG_PATH", mock_path):
            result = mgr._load_sent_today()
        assert result == set()

    def test_load_sent_today_from_csv(self):
        """_load_sent_today returns only today's (ticker, signal_name) pairs."""
        mgr    = _make_mgr()
        today  = date.today().isoformat()
        mgr._today_str = today

        fake_df = pd.DataFrame({
            "date":        [today,   today,   "2000-01-01"],
            "ticker":      ["AAPL",  "MSFT",  "GOOGL"],
            "signal_name": ["LONG",  "SHORT", "LONG"],
            "confidence":  ["0.75",  "0.80",  "0.70"],
            "channel":     ["email", "telegram", "email"],
        })
        mock_path = MagicMock()
        mock_path.exists.return_value = True

        with patch("alerts.notifier._LOG_PATH", mock_path):
            with patch("alerts.notifier.pd.read_csv", return_value=fake_df):
                result = mgr._load_sent_today()

        assert ("AAPL",  "LONG")  in result
        assert ("MSFT",  "SHORT") in result
        assert ("GOOGL", "LONG")  not in result   # from yesterday
        assert len(result) == 2
