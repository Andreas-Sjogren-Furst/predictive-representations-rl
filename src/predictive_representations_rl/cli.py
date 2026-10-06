from __future__ import annotations

import argparse
import sys
from pathlib import Path

from predictive_representations_rl.core import registry
from predictive_representations_rl.core.compat import check
from predictive_representations_rl.core.config import load_config
from predictive_representations_rl.core.runtime import find_project_root


def _list(what: str) -> int:
    if what == "envs":
        for env in registry.ENVS.values():
            datasets = ", ".join(env.datasets) or "-"
            print(f"{env.name}  [{env.suite}, {env.action_type} actions, obs: {'/'.join(sorted(env.observation_types))}]")
            print(f"    {env.description}")
            print(f"    tasks: {', '.join(env.tasks)}")
            print(f"    datasets: {datasets}")
            print(f"    algorithms: {', '.join(env.native_names)}")
    else:
        for algo in registry.ALGORITHMS.values():
            modes = ", ".join(f"{mode} ({requirement})" for mode, requirement in algo.modes.items())
            print(f"{algo.name}")
            print(f"    {algo.description}")
            print(f"    modes: {modes}")
            print(f"    actions: {'/'.join(sorted(algo.action_spaces))}  obs: {'/'.join(sorted(algo.observation_types))}")
            print(f"    training: {'single task' if algo.task_specific else 'task-agnostic pretraining'}")
            stateful = " (history-dependent)" if algo.stateful_representation else ""
            print(f"    representations: {', '.join(algo.representations)}{stateful}")
    return 0


def _check(paths: list[Path]) -> int:
    root = find_project_root()
    failed = False

    for path in paths:
        try:
            config = load_config(path)
            env = registry.get_env(config.env)
            algo = registry.get_algorithm(config.algo)
        except (KeyError, ValueError, OSError) as error:
            print(f"✗ {path}: {error}")
            failed = True
            continue

        issues = check(config, env, algo, root)
        label = f"{path}  ({algo.name} · {env.name} · {config.mode} · {config.obs_type})"

        if not issues:
            print(f"✓ {label}")
            continue

        failed = True
        print(f"✗ {label}")
        for issue in issues:
            print(f"    [{issue.kind}] {issue.message}")

    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prl", description="Predictive-representations experiment harness.")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List registered environments or algorithms.")
    list_parser.add_argument("what", choices=["envs", "algos"])

    check_parser = commands.add_parser("check", help="Check experiment configs without running them.")
    check_parser.add_argument("configs", nargs="+", type=Path)

    args = parser.parse_args(argv)

    if args.command == "list":
        return _list(args.what)
    return _check(args.configs)


if __name__ == "__main__":
    sys.exit(main())
