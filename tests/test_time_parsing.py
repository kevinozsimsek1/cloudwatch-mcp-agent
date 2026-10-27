from datetime import datetime, timezone

import pytest

from app.sre_context import calendar_window, parse_hours_from_message, parse_time_window

# 2026-10-27 13:46 Europe/Istanbul
NOW = datetime(2026, 10, 27, 10, 46, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("message", "hours"),
    [
        ("son 6 saat", 6),
        ("son 6 saatte 500 var mı", 6),
        ("son 12 saatde kaç hata almış", 12),
        ("son 3 saattir", 3),
        ("son 24 saatlik özet", 24),
        ("son 2 günde", 48),
        ("son 1 haftada", 168),
        ("last 3 hours", 3),
        ("24h", 24),
        ("son iki saatte", 2),
        ("on iki saat önce başladı", 12),
    ],
)
def test_turkish_and_english_hour_phrases(message, hours):
    assert parse_hours_from_message(message) == hours


@pytest.mark.parametrize(
    ("message", "minutes"),
    [
        ("son 30 dakikada 500 var mı", 30),
        ("son 45 dk", 45),
        ("last 15 minutes", 15),
        ("son yarım saat", 30),
        ("30m", 30),
    ],
)
def test_minute_windows_keep_minute_precision(message, minutes):
    window = parse_time_window(message, NOW)
    assert window is not None
    assert window.minutes == minutes
    assert window.is_sub_hour_precision
    assert window.label == f"son {minutes} dakika"


@pytest.mark.parametrize(
    "message",
    ["500 var mı?", "status kodu 500 olan kaç log grubum var", "ec2h", "/aws/lambda/v2d-handler"],
)
def test_no_time_phrase(message):
    assert parse_time_window(message, NOW) is None
    assert parse_hours_from_message(message) == 1


def test_yesterday_is_istanbul_calendar_day():
    window = parse_time_window("Dün kaç tane 500 hatası oldu?", NOW)
    assert window.is_fixed
    assert window.start == datetime(2026, 10, 25, 21, 0, tzinfo=timezone.utc)
    assert window.end == datetime(2026, 10, 26, 21, 0, tzinfo=timezone.utc)
    assert window.hours == 24
    assert "26.10.2026" in window.label


def test_today_starts_at_istanbul_midnight():
    window = calendar_window("today", NOW)
    assert window.start == datetime(2026, 10, 26, 21, 0, tzinfo=timezone.utc)
    assert window.end == NOW
    assert window.hours == 14


def test_last_week_is_previous_monday_to_sunday():
    window = calendar_window("last_week", NOW)
    assert window.start == datetime(2026, 10, 18, 21, 0, tzinfo=timezone.utc)
    assert window.end == datetime(2026, 10, 25, 21, 0, tzinfo=timezone.utc)


def test_explicit_number_beats_calendar_word():
    window = parse_time_window("dün son 3 saatte ne oldu", NOW)
    assert not window.is_fixed
    assert window.hours == 3
