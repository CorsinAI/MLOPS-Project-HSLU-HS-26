"""Tests for central configuration."""

from neghour import config


def test_cutoff_defaults_to_ten_in_berlin() -> None:
    assert config.CUTOFF_HOUR == 10
    assert config.BERLIN_TZ.key == "Europe/Berlin"
