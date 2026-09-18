"""Compare policy-learning and fine-tuning costs without contacting cloud services."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any


def nonnegative_decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"{name} must be a finite non-negative number")
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class LinearCost:
    fixed: Decimal
    per_episode: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixed", nonnegative_decimal(self.fixed, "fixed"))
        object.__setattr__(self, "per_episode", nonnegative_decimal(self.per_episode, "per_episode"))

    def total(self, episodes: int) -> Decimal:
        return self.fixed + self.per_episode * nonnegative_integer(episodes, "episodes")


def break_even(policy: LinearCost, fine_tuning: LinearCost) -> dict[str, object]:
    """Find a continuous-volume crossover, holding cadence and unit prices fixed."""
    fixed_difference = policy.fixed - fine_tuning.fixed
    variable_difference = policy.per_episode - fine_tuning.per_episode
    if variable_difference == 0:
        if fixed_difference == 0:
            return {"status": "equal_at_all_volumes"}
        return {
            "status": "no_crossover",
            "cheaper": "policy_learning" if fixed_difference < 0 else "fine_tuning",
        }

    crossover = -fixed_difference / variable_difference
    if crossover < 0:
        return {
            "status": "no_crossover",
            "cheaper": "policy_learning" if fixed_difference < 0 else "fine_tuning",
        }

    cheaper_above = "policy_learning" if variable_difference < 0 else "fine_tuning"
    return {
        "status": "crossover",
        "episode_volume": crossover,
        "cheaper_below": (
            ("fine_tuning" if cheaper_above == "policy_learning" else "policy_learning")
            if crossover > 0
            else None
        ),
        "cheaper_above": cheaper_above,
        "first_integer_volume_above": int(crossover.to_integral_value(rounding=ROUND_FLOOR)) + 1,
    }


def validate_keys(payload: object, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be an object")
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(sorted(unknown))}")
    return payload


@dataclass(frozen=True)
class Approach:
    evaluation_fraction: Decimal = Decimal(1)
    judge_calls_per_evaluated_episode: Decimal = Decimal(0)
    judge_cost_per_call: Decimal = Decimal(0)
    cpu_seconds_per_episode: Decimal = Decimal(0)
    cpu_cost_per_hour: Decimal = Decimal(0)
    writes_per_episode: Decimal = Decimal(0)
    cost_per_million_writes: Decimal = Decimal(0)
    average_retained_gb: Decimal = Decimal(0)
    average_retained_gb_per_episode: Decimal = Decimal(0)
    storage_cost_per_gb_month: Decimal = Decimal(0)
    cycles_per_period: int = 0
    compute_cost_per_cycle: Decimal = Decimal(0)
    rollout_cost_per_cycle: Decimal = Decimal(0)
    fixed_cost_per_period: Decimal = Decimal(0)
    serving_cost_per_episode: Decimal = Decimal(0)
    other_cost_per_episode: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        for attribute in fields(self):
            value = getattr(self, attribute.name)
            validate = nonnegative_integer if attribute.name == "cycles_per_period" else nonnegative_decimal
            object.__setattr__(self, attribute.name, validate(value, attribute.name))
        if self.evaluation_fraction > 1:
            raise ValueError("evaluation_fraction must be between 0 and 1")

    @classmethod
    def from_dict(cls, payload: object, name: str) -> Approach:
        return cls(**validate_keys(payload, {attribute.name for attribute in fields(cls)}, name))

    def components(self, period_days: int) -> dict[str, LinearCost]:
        storage_rate = self.storage_cost_per_gb_month * Decimal(period_days) / 30
        return {
            "judge_evaluation": LinearCost(
                Decimal(0),
                self.evaluation_fraction * self.judge_calls_per_evaluated_episode * self.judge_cost_per_call,
            ),
            "episode_cpu": LinearCost(
                Decimal(0),
                self.cpu_seconds_per_episode * self.cpu_cost_per_hour / 3600,
            ),
            "storage_writes": LinearCost(
                Decimal(0),
                self.writes_per_episode * self.cost_per_million_writes / 1_000_000,
            ),
            "storage_capacity": LinearCost(
                self.average_retained_gb * storage_rate,
                self.average_retained_gb_per_episode * storage_rate,
            ),
            "cycle_compute": LinearCost(self.cycles_per_period * self.compute_cost_per_cycle, Decimal(0)),
            "rollouts": LinearCost(self.cycles_per_period * self.rollout_cost_per_cycle, Decimal(0)),
            "other_fixed": LinearCost(self.fixed_cost_per_period, Decimal(0)),
            "serving": LinearCost(Decimal(0), self.serving_cost_per_episode),
            "other_variable": LinearCost(Decimal(0), self.other_cost_per_episode),
        }


def sum_costs(components: dict[str, LinearCost]) -> LinearCost:
    return LinearCost(
        sum((component.fixed for component in components.values()), Decimal(0)),
        sum((component.per_episode for component in components.values()), Decimal(0)),
    )


@dataclass(frozen=True)
class Scenario:
    currency: str
    period_days: int
    episodes: int
    assumptions: tuple[str, ...]
    shared: LinearCost
    policy_learning: Approach
    fine_tuning: Approach

    @classmethod
    def from_dict(cls, payload: object) -> Scenario:
        names = {attribute.name for attribute in fields(cls)}
        data = validate_keys(payload, names, "scenario")
        missing = names - set(data)
        if missing:
            raise ValueError(f"scenario is missing fields: {', '.join(sorted(missing))}")
        if not isinstance(data["currency"], str) or not data["currency"].strip():
            raise ValueError("currency must be a non-empty string")
        period_days = nonnegative_integer(data["period_days"], "period_days")
        if period_days == 0:
            raise ValueError("period_days must be greater than zero")
        assumptions = data["assumptions"]
        if (
            not isinstance(assumptions, list)
            or not assumptions
            or any(not isinstance(item, str) or not item.strip() for item in assumptions)
        ):
            raise ValueError("assumptions must be a non-empty list of non-empty strings")
        shared = validate_keys(data["shared"], {"fixed", "per_episode"}, "shared")
        if set(shared) != {"fixed", "per_episode"}:
            raise ValueError("shared must specify fixed and per_episode costs, including explicit zeros")
        return cls(
            currency=data["currency"].strip(),
            period_days=period_days,
            episodes=nonnegative_integer(data["episodes"], "episodes"),
            assumptions=tuple(assumptions),
            shared=LinearCost(**shared),
            policy_learning=Approach.from_dict(data["policy_learning"], "policy_learning"),
            fine_tuning=Approach.from_dict(data["fine_tuning"], "fine_tuning"),
        )

    def compare(self, volumes: list[int] | None = None) -> dict[str, object]:
        if volumes is not None and not volumes:
            raise ValueError("volumes must contain at least one episode count")
        approaches = {
            "policy_learning": {**self.policy_learning.components(self.period_days), "shared": self.shared},
            "fine_tuning": {**self.fine_tuning.components(self.period_days), "shared": self.shared},
        }
        totals = {name: sum_costs(components) for name, components in approaches.items()}
        estimates = []
        for episodes in [self.episodes] if volumes is None else volumes:
            nonnegative_integer(episodes, "episodes")
            difference = totals["fine_tuning"].total(episodes) - totals["policy_learning"].total(episodes)
            estimates.append(
                {
                    "episodes": episodes,
                    **{
                        name: {
                            "fixed": totals[name].fixed,
                            "per_episode": totals[name].per_episode,
                            "costs": {
                                label: component.total(episodes) for label, component in components.items()
                            },
                            "total": totals[name].total(episodes),
                        }
                        for name, components in approaches.items()
                    },
                    "fine_tuning_minus_policy_learning": difference,
                    "cheaper": (
                        "policy_learning" if difference > 0 else "fine_tuning" if difference < 0 else "equal"
                    ),
                }
            )
        return {
            "currency": self.currency,
            "period_days": self.period_days,
            "assumptions": self.assumptions,
            "break_even": break_even(totals["policy_learning"], totals["fine_tuning"]),
            "estimates": estimates,
        }


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def episode_argument(value: str) -> int:
    try:
        return nonnegative_integer(int(value), "episodes")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("episodes must be a non-negative integer") from exc


def serialize_decimal(value: object) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="JSON scenario; all prices supplied by you"
    )
    parser.add_argument(
        "--episodes", nargs="+", type=episode_argument, help="Override volume or run a volume sweep"
    )
    args = parser.parse_args(argv)
    try:
        with args.config.open(encoding="utf-8") as stream:
            data = json.load(
                stream, parse_float=Decimal, parse_constant=Decimal, object_pairs_hook=unique_object
            )
        report = Scenario.from_dict(data).compare(args.episodes)
    except (OSError, UnicodeError, TypeError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, default=serialize_decimal, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
