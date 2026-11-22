"""The dashboard shows amounts in the currency of the data being served, not always rupees."""

import json

import pytest

from backend.config.settings import resolve_currency_symbol


def workspace(tmp_path, currency):
    ws = tmp_path / "sparkov"
    (ws / "data_store").mkdir(parents=True, exist_ok=True)
    (ws / "config.json").write_text(json.dumps({"dataset": "sparkov", "currency": currency}))
    return str(ws / "data_store")


def test_the_default_is_rupees_as_in_the_original_synthetic_data(tmp_path):
    assert resolve_currency_symbol(None, str(tmp_path / "ml" / "data_store")) == "₹"


@pytest.mark.parametrize("code,symbol", [("USD", "$"), ("INR", "₹"), ("EUR", "€"), ("GBP", "£")])
def test_a_workspace_reports_its_datasets_currency(tmp_path, code, symbol):
    assert resolve_currency_symbol(None, workspace(tmp_path, code)) == symbol


def test_an_explicit_setting_beats_the_workspace(tmp_path):
    assert resolve_currency_symbol("¥", workspace(tmp_path, "USD")) == "¥"


def test_an_unknown_or_missing_currency_falls_back_to_rupees(tmp_path):
    assert resolve_currency_symbol(None, workspace(tmp_path, "unspecified")) == "₹"
    assert resolve_currency_symbol(None, workspace(tmp_path, None)) == "₹"


def test_an_unreadable_workspace_config_does_not_break_startup(tmp_path):
    ds = workspace(tmp_path, "USD")
    (tmp_path / "sparkov" / "config.json").write_text("not json")
    assert resolve_currency_symbol(None, ds) == "₹"


def test_health_reports_the_currency(client):
    body = client.get("/health").json()
    assert body["currency"] in {"₹", "$", "€", "£", "¥"} or len(body["currency"]) <= 3
