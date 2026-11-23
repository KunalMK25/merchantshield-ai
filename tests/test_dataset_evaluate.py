"""Evaluating models and the account detector on a prepared dataset's real labels."""

import json

import pytest

from ml.datasets.evaluate import evaluate_workspace, main
from ml.datasets.prepare import prepare
from ml.datasets.workspace import Workspace
from tests.dataset_fixtures import write_sparkov_like

# Tiny fixtures can leave a validation window with no fraud rows; sklearn warns, the code handles it.
pytestmark = pytest.mark.filterwarnings("ignore:No positive class found:UserWarning")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root, src = tmp_path_factory.mktemp("ws"), tmp_path_factory.mktemp("src")
    write_sparkov_like(src / "fraudTrain.csv", customers=80, days=80)
    prepare("sparkov", src, root, stream_days=15)
    return Workspace("sparkov", root)


@pytest.fixture(scope="module")
def report(prepared):
    return evaluate_workspace(prepared, models=["logistic_regression", "random_forest"])


def test_the_report_covers_data_fit_split_and_every_model(report):
    assert report["dataset"] == "sparkov" and report["fit"] == "good"
    assert report["rows"] > 0 and report["fraud_rate"] > 0
    assert {"train", "val", "test"} == set(report["split"]) and all(v > 0 for v in report["split"].values())
    names = {m["name"] for m in report["contest"]}
    assert names == {"logistic_regression", "random_forest"}
    for m in report["contest"]:
        assert 0 <= m["val"]["precision"] <= 1 and 0 <= m["test"]["recall"] <= 1 and m["test"]["cost"] >= 0
    assert report["winner"] in names


def test_it_uses_the_workspace_cost_scale_and_states_it(report, prepared):
    cfg = json.loads(prepared.config_path.read_text())
    assert report["cost"]["fp_cost"] == cfg["fp_cost"]
    assert "assumed" in report["cost_note"].lower()


def test_it_reports_how_the_account_detector_does_on_real_labels(report):
    a = report["anomaly"]
    assert set(a["tiers"]) == {"critical", "critical_plus_high", "everything_flagged"}
    for tier in a["tiers"].values():
        assert tier["flagged"] >= 0 and (tier["precision"] is None or 0 <= tier["precision"] <= 1)
    assert a["compromised_account_days"] >= 0 and "accounts_checked" in a


def test_the_split_holds_out_the_most_recent_days(prepared):
    r = evaluate_workspace(prepared, models=["logistic_regression"], val_days=5, test_days=5)
    assert r["val_days"] == 5 and r["test_days"] == 5


def test_it_writes_the_report_and_the_command_prints_a_summary(prepared, capsys):
    main(["sparkov", "--workspace-root", str(prepared.root), "--models", "logistic_regression", "random_forest"])
    out = capsys.readouterr().out
    assert "logistic_regression" in out and "random_forest" in out and "precision" in out.lower()
    def reject(token):
        raise ValueError(f"non-standard JSON constant {token}")

    saved = json.loads((prepared.path / "evaluation.json").read_text(), parse_constant=reject)
    assert saved["dataset"] == "sparkov"


def test_an_unprepared_workspace_is_a_clear_error(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["sparkov", "--workspace-root", str(tmp_path)])
    assert "prepare" in capsys.readouterr().out


def test_a_dataset_too_short_for_the_windows_is_a_clear_error(prepared, capsys):
    """Validation plus test windows that leave no training days used to end in an obscure scikit-learn error."""
    with pytest.raises(ValueError, match="spans only"):
        evaluate_workspace(prepared, models=["logistic_regression"], val_days=60, test_days=60)
    with pytest.raises(SystemExit):
        main(["sparkov", "--workspace-root", str(prepared.root), "--val-days", "60", "--test-days", "60"])
    assert "spans only" in capsys.readouterr().out
