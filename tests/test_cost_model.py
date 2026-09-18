"""Offline tests for the standalone cost comparison script."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "estimate_learning_cost.py"
SPEC = importlib.util.spec_from_file_location("estimate_learning_cost", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cost_model = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cost_model
SPEC.loader.exec_module(cost_model)


def test_linear_total_includes_fixed_cost_at_zero_volume() -> None:
    cost = cost_model.LinearCost(Decimal(10), Decimal("0.02"))
    assert cost.total(0) == Decimal(10)
    assert cost.total(1000) == Decimal(30)


@pytest.mark.parametrize("reverse", [False, True])
def test_crossover_matches_totals_and_both_directions(reverse: bool) -> None:
    policy = cost_model.LinearCost(Decimal(10), Decimal("0.03"))
    fine_tuning = cost_model.LinearCost(Decimal(30), Decimal("0.01"))
    if reverse:
        policy, fine_tuning = fine_tuning, policy
    result = cost_model.break_even(policy, fine_tuning)
    assert result["status"] == "crossover"
    assert result["episode_volume"] == Decimal(1000)
    assert result["first_integer_volume_above"] == 1001
    assert result["cheaper_below"] == ("fine_tuning" if reverse else "policy_learning")
    assert result["cheaper_above"] == ("policy_learning" if reverse else "fine_tuning")
    assert policy.total(1000) == fine_tuning.total(1000) == Decimal(40)
    assert (policy.total(1001) < fine_tuning.total(1001)) is reverse


@pytest.mark.parametrize("fine_rate", ["0.01", "0.02"])
def test_no_crossover_when_policy_is_always_cheaper(fine_rate: str) -> None:
    policy = cost_model.LinearCost(Decimal(10), Decimal("0.01"))
    fine_tuning = cost_model.LinearCost(Decimal(30), Decimal(fine_rate))
    assert cost_model.break_even(policy, fine_tuning) == {
        "status": "no_crossover",
        "cheaper": "policy_learning",
    }
    assert cost_model.break_even(fine_tuning, policy) == {
        "status": "no_crossover",
        "cheaper": "fine_tuning",
    }


def test_identical_costs_are_equal_at_all_volumes() -> None:
    cost = cost_model.LinearCost(Decimal(10), Decimal("0.01"))
    assert cost_model.break_even(cost, cost) == {"status": "equal_at_all_volumes"}


@pytest.mark.parametrize("fixed, expected", [("0", "0"), ("1", "0.5")])
def test_zero_and_fractional_crossovers(fixed: str, expected: str) -> None:
    result = cost_model.break_even(
        cost_model.LinearCost(Decimal(0), Decimal(2)),
        cost_model.LinearCost(Decimal(fixed), Decimal(0)),
    )
    assert result["episode_volume"] == Decimal(expected)
    assert result["first_integer_volume_above"] == 1
    if fixed == "0":
        assert result["cheaper_below"] is None


@pytest.mark.parametrize(
    "invalid, expected_error",
    [
        (-1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("1", TypeError),
        (None, TypeError),
    ],
)
def test_reject_invalid_costs(invalid: object, expected_error: type[Exception]) -> None:
    with pytest.raises(expected_error, match="finite non-negative") as error:
        cost_model.LinearCost(invalid, Decimal(0))
    assert type(error.value) is expected_error


@pytest.mark.parametrize(
    "invalid, expected_error",
    [(-1, ValueError), (1.5, TypeError), (True, TypeError), ("1", TypeError), (None, TypeError)],
)
def test_reject_invalid_episode_counts(invalid: object, expected_error: type[Exception]) -> None:
    with pytest.raises(expected_error, match="non-negative integer") as error:
        cost_model.LinearCost(Decimal(0), Decimal(0)).total(invalid)
    assert type(error.value) is expected_error


def scenario_payload() -> dict:
    return {
        "currency": "USD",
        "period_days": 30,
        "episodes": 1000,
        "assumptions": ["Illustrative prices, not provider quotes"],
        "shared": {"fixed": 100, "per_episode": 0.001},
        "policy_learning": {
            "evaluation_fraction": 0.5,
            "judge_calls_per_evaluated_episode": 3,
            "judge_cost_per_call": 0.002,
            "cpu_seconds_per_episode": 1,
            "cpu_cost_per_hour": 0.36,
            "writes_per_episode": 8,
            "cost_per_million_writes": 2,
            "average_retained_gb": 5,
            "average_retained_gb_per_episode": 0.00001,
            "storage_cost_per_gb_month": 0.2,
            "cycles_per_period": 4,
            "compute_cost_per_cycle": 0.5,
            "rollout_cost_per_cycle": 2,
            "fixed_cost_per_period": 10,
            "serving_cost_per_episode": 0.01,
            "other_cost_per_episode": 0.001,
        },
        "fine_tuning": {"cycles_per_period": 2, "compute_cost_per_cycle": 50},
    }


def test_cost_driver_breakdown_matches_hand_calculation() -> None:
    estimate = cost_model.Scenario.from_dict(scenario_payload()).compare()["estimates"][0]
    policy = estimate["policy_learning"]
    assert policy["costs"] == {
        "judge_evaluation": Decimal(3),
        "episode_cpu": Decimal("0.1"),
        "storage_writes": Decimal("0.016"),
        "storage_capacity": Decimal("1.002"),
        "cycle_compute": Decimal(2),
        "rollouts": Decimal(8),
        "other_fixed": Decimal(10),
        "serving": Decimal(10),
        "other_variable": Decimal(1),
        "shared": Decimal(101),
    }
    assert policy["total"] == Decimal("136.118") == sum(policy["costs"].values())
    assert estimate["fine_tuning"]["total"] == Decimal(201)
    assert estimate["fine_tuning_minus_policy_learning"] == Decimal("64.882")
    assert estimate["cheaper"] == "policy_learning"


def test_shared_costs_change_totals_but_not_savings_or_crossover() -> None:
    payload = scenario_payload()
    with_shared = cost_model.Scenario.from_dict(payload).compare()
    payload["shared"] = {"fixed": 0, "per_episode": 0}
    without_shared = cost_model.Scenario.from_dict(payload).compare()
    assert with_shared["break_even"] == without_shared["break_even"]
    first = with_shared["estimates"][0]
    second = without_shared["estimates"][0]
    assert first["fine_tuning_minus_policy_learning"] == second["fine_tuning_minus_policy_learning"]
    assert first["fine_tuning"]["total"] - second["fine_tuning"]["total"] == Decimal(101)


def test_zero_volume_keeps_scheduled_cycles_and_existing_storage() -> None:
    estimate = cost_model.Scenario.from_dict(scenario_payload()).compare([0])["estimates"][0]
    assert estimate["policy_learning"]["total"] == Decimal(121)
    assert estimate["fine_tuning"]["total"] == Decimal(200)


def test_compare_rejects_empty_sweep() -> None:
    scenario = cost_model.Scenario.from_dict(scenario_payload())
    with pytest.raises(ValueError, match="volumes must contain at least one episode count"):
        scenario.compare([])


def test_compare_none_uses_configured_volume() -> None:
    scenario = cost_model.Scenario.from_dict(scenario_payload())
    assert scenario.compare(None) == scenario.compare([1000])


def test_storage_prorates_to_period_and_local_scoring_can_cost_cpu() -> None:
    payload = scenario_payload()
    payload["period_days"] = 15
    payload["policy_learning"]["judge_cost_per_call"] = 0
    estimate = cost_model.Scenario.from_dict(payload).compare()["estimates"][0]
    assert estimate["policy_learning"]["costs"]["judge_evaluation"] == 0
    assert estimate["policy_learning"]["costs"]["episode_cpu"] == Decimal("0.1")
    assert estimate["policy_learning"]["costs"]["storage_capacity"] == Decimal("0.501")


@pytest.mark.parametrize(
    "field, invalid, expected_error",
    [
        ("evaluation_fraction", 1.1, ValueError),
        ("evaluation_fraction", -0.1, ValueError),
        ("cycles_per_period", 0.5, TypeError),
        ("cycles_per_period", True, TypeError),
        ("judge_cost_per_call", float("nan"), ValueError),
        ("cpu_cost_per_hour", float("inf"), ValueError),
        ("writes_per_episode", -1, ValueError),
        ("fixed_cost_per_period", "12", TypeError),
        ("judge_price_typo", 1, ValueError),
    ],
)
def test_reject_invalid_approach_inputs(
    field: str, invalid: object, expected_error: type[Exception]
) -> None:
    payload = scenario_payload()
    payload["policy_learning"][field] = invalid
    with pytest.raises(expected_error) as error:
        cost_model.Scenario.from_dict(payload)
    assert type(error.value) is expected_error


@pytest.mark.parametrize(
    "field, invalid, expected_error",
    [
        ("currency", "", ValueError),
        ("currency", None, ValueError),
        ("period_days", 0, ValueError),
        ("period_days", 1.5, TypeError),
        ("episodes", -1, ValueError),
        ("episodes", True, TypeError),
        ("assumptions", [], ValueError),
        ("assumptions", "assumed", ValueError),
        ("assumptions", [""], ValueError),
        ("shared", {}, ValueError),
        ("shared", {"fixed": 0, "per_episode": -1}, ValueError),
        ("policy_learning", [], TypeError),
        ("unknown", 0, ValueError),
    ],
)
def test_reject_invalid_scenarios(field: str, invalid: object, expected_error: type[Exception]) -> None:
    payload = scenario_payload()
    payload[field] = invalid
    with pytest.raises(expected_error) as error:
        cost_model.Scenario.from_dict(payload)
    assert type(error.value) is expected_error


def test_require_both_approaches_and_common_period() -> None:
    payload = scenario_payload()
    del payload["fine_tuning"]
    with pytest.raises(ValueError, match="missing fields: fine_tuning"):
        cost_model.Scenario.from_dict(payload)


def test_cli_runs_outside_repository_and_outputs_json(tmp_path: Path) -> None:
    config = tmp_path / "scenario.json"
    config.write_text(json.dumps(scenario_payload()), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--config", str(config), "--episodes", "0", "1000"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["currency"] == "USD"
    assert [row["episodes"] for row in report["estimates"]] == [0, 1000]
    assert Decimal(report["estimates"][1]["policy_learning"]["total"]) == Decimal("136.118")
    assert result.stderr == ""


@pytest.mark.parametrize("content", ["{", "[]", '{"episodes": 1, "episodes": 2}'])
def test_cli_rejects_bad_json_without_traceback(tmp_path: Path, content: str) -> None:
    config = tmp_path / "invalid.json"
    config.write_text(content, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_cli_rejects_missing_file(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(tmp_path / "missing.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("volume", ["-1", "1.5", "nan"])
def test_cli_rejects_invalid_volume(volume: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "scenario.json"
    config.write_text(json.dumps(scenario_payload()), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cost_model.main(["--config", str(config), "--episodes", volume])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "argument --episodes: episodes must be a non-negative integer" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_accepts_valid_volume(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "scenario.json"
    config.write_text(json.dumps(scenario_payload()), encoding="utf-8")
    assert cost_model.main(["--config", str(config), "--episodes", "10"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert [row["episodes"] for row in report["estimates"]] == [10]
    assert report["estimates"][0]["policy_learning"]["total"] == "121.151180"
    assert captured.err == ""


def test_cli_rejects_unexpected_report_objects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "scenario.json"
    config.write_text(json.dumps(scenario_payload()), encoding="utf-8")
    monkeypatch.setattr(cost_model.Scenario, "compare", lambda self, volumes: {"unexpected": object()})
    with pytest.raises(TypeError, match="not JSON serializable"):
        cost_model.main(["--config", str(config)])


@pytest.mark.parametrize(
    "scenario, policy_totals",
    [
        ("local", ["120.50", "140.65", "160.80", "322.00"]),
        ("hosted", ["120.50", "260.65", "400.80", "1522.00"]),
    ],
)
def test_documented_scenarios_reproduce_worked_table(scenario: str, policy_totals: list[str]) -> None:
    config = SCRIPT.parents[1] / "examples" / f"cost_model_{scenario}.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            "--config",
            str(config),
            "--episodes",
            "0",
            "10000",
            "20000",
            "100000",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["period_days"] == 30
    assert report["assumptions"]
    fine_totals = ["330.50", "362.65", "394.80", "652.00"]
    for row, policy, fine in zip(report["estimates"], policy_totals, fine_totals):
        assert Decimal(row["policy_learning"]["total"]) == Decimal(policy)
        assert Decimal(row["fine_tuning"]["total"]) == Decimal(fine)
        assert sum(Decimal(value) for value in row["policy_learning"]["costs"].values()) == Decimal(policy)
    if scenario == "local":
        assert report["break_even"] == {"status": "no_crossover", "cheaper": "policy_learning"}
    else:
        assert report["break_even"]["status"] == "crossover"
        assert report["break_even"]["first_integer_volume_above"] == 19445
        assert report["break_even"]["cheaper_above"] == "fine_tuning"
        assert abs(
            Decimal(report["break_even"]["episode_volume"]) - Decimal("19444.44444444444444444444444")
        ) <= Decimal("1e-20")


def test_hosted_scenario_switches_winner_at_first_integer_above_crossover() -> None:
    config = SCRIPT.parents[1] / "examples" / "cost_model_hosted.json"
    scenario = cost_model.Scenario.from_dict(
        json.loads(config.read_text(encoding="utf-8"), parse_float=Decimal)
    )
    rows = scenario.compare([19444, 19445])["estimates"]
    assert rows[0]["cheaper"] == "policy_learning"
    assert rows[1]["cheaper"] == "fine_tuning"


def test_equal_evaluation_coverage_removes_hosted_crossover() -> None:
    config = SCRIPT.parents[1] / "examples" / "cost_model_hosted.json"
    payload = json.loads(config.read_text(encoding="utf-8"), parse_float=Decimal)
    payload["fine_tuning"]["evaluation_fraction"] = 1
    report = cost_model.Scenario.from_dict(payload).compare([0, 10000, 100000])
    assert report["break_even"] == {"status": "no_crossover", "cheaper": "policy_learning"}
    assert all(row["fine_tuning_minus_policy_learning"] == Decimal(210) for row in report["estimates"])


def test_fine_tuned_serving_premium_moves_hosted_crossover() -> None:
    config = SCRIPT.parents[1] / "examples" / "cost_model_hosted.json"
    payload = json.loads(config.read_text(encoding="utf-8"), parse_float=Decimal)
    payload["fine_tuning"]["serving_cost_per_episode"] = Decimal("0.0024")
    report = cost_model.Scenario.from_dict(payload).compare([24999, 25000, 25001])
    assert report["break_even"] == {
        "status": "crossover",
        "episode_volume": Decimal(25000),
        "cheaper_below": "policy_learning",
        "cheaper_above": "fine_tuning",
        "first_integer_volume_above": 25001,
    }
    rows = report["estimates"]
    assert [row["cheaper"] for row in rows] == ["policy_learning", "equal", "fine_tuning"]
    assert rows[1]["fine_tuning"]["costs"]["serving"] == Decimal(60)
    assert rows[1]["policy_learning"]["total"] == Decimal("470.875")
    assert rows[1]["fine_tuning"]["total"] == Decimal("470.875")


@pytest.mark.parametrize("episodes, expected_message", [
    (True, "episodes must be a non-negative integer"),
    (-1, "episodes must be a non-negative integer"),
])
def test_cli_reports_invalid_config_numbers(
    episodes: object, expected_message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "scenario.json"
    payload = scenario_payload()
    payload["episodes"] = episodes
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cost_model.main(["--config", str(config)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert expected_message in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
