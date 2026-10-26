import pytest

from aeris.analysis.goal_policy import build_goal_oracle
from aeris.data import RunDirectory, write_transitions_npz
from aeris.domain import PolicySnapshot, TabularPolicyReport
from aeris.environments import FourRooms
from aeris.workflows.four_rooms import (
    analyze_saved_policies,
    diagnose_fb_checkpoint,
    open_four_rooms_run,
    render_policy_figures,
    render_run_figures,
)


def with_optimal_snapshot(run):
    oracle = build_goal_oracle(FourRooms().tabular_dynamics(), (9, 9), gamma=0.99)
    run.write_policy_snapshot(
        "final", PolicySnapshot(states=oracle.states, actions=oracle.optimal_policy())
    )
    return run


def test_open_reads_goal_and_horizon(four_rooms_run):
    run = open_four_rooms_run(four_rooms_run())
    assert (run.goal, run.max_steps, run.task.task_id) == ((9, 9), 10, "reach_9_9")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"status": "running"}, "completed"),
        ({"environment": "other_env"}, "only Four Rooms"),
    ],
)
def test_ineligible_runs_are_refused(four_rooms_run, overrides, message):
    with pytest.raises(ValueError, match=message):
        open_four_rooms_run(four_rooms_run(**overrides))


def test_policy_analysis_saves_the_report(four_rooms_run):
    run = with_optimal_snapshot(four_rooms_run())
    analysis = analyze_saved_policies(run)
    assert analysis.report.snapshots["final"].optimal_action_fraction == 1.0
    assert analysis.report.snapshots["final"].policy_snapshot == "policy_final.npz"
    assert run.read_report(TabularPolicyReport) == analysis.report


def test_policy_analysis_needs_a_snapshot(four_rooms_run):
    with pytest.raises(ValueError, match="no saved policy snapshots"):
        analyze_saved_policies(four_rooms_run())


def test_out_of_range_snapshot_actions_are_refused(four_rooms_run):
    run = four_rooms_run()
    env = FourRooms()
    run.write_policy_snapshot(
        "final", PolicySnapshot(states=env.positions, actions=[10] * len(env.positions))
    )
    with pytest.raises(ValueError, match="invalid actions"):
        analyze_saved_policies(run)


def test_policy_figures(four_rooms_run):
    pytest.importorskip("matplotlib")
    run = with_optimal_snapshot(four_rooms_run())
    made = render_policy_figures(run, analyze_saved_policies(run))
    assert [p.name for p in made] == ["optimal_policy.png", "policy_gap_final.png"]
    assert [p.name for p in render_run_figures(run)] == ["policy_final.png"]


class TestFBCheckpoint:
    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")
        self.torch.set_num_threads(1)

    def make_run(self, tmp_path, four_rooms_run, *, origin="oracle", transitions=None):
        """An oracle-dataset run with an untrained FB checkpoint, as training saves it."""
        from aeris.agents.fb import DiscreteFBAgent, DiscreteFBConfig
        from aeris.agents.fb.checkpoint import save_checkpoint

        env = FourRooms()
        dataset = RunDirectory(tmp_path / "run").dataset_path
        fingerprint = write_transitions_npz(dataset, transitions or tuple(env.all_transitions()))
        run = four_rooms_run(
            max_steps=15, dataset_settings={"origin": origin, "sha256": fingerprint}
        )
        config = DiscreteFBConfig(
            observation_dim=env.observation_dim,
            num_actions=env.num_actions,
            representation_dim=6,
            hidden_dim=10,
            seed=123,
        )
        save_checkpoint(run.checkpoint_path, DiscreteFBAgent(config))
        return run

    def test_diagnostic_saves_report_and_leaves_checkpoint_untouched(
        self, tmp_path, four_rooms_run
    ):
        from aeris.agents.fb import FBCheckpointReport

        run = self.make_run(tmp_path, four_rooms_run)
        original = run.checkpoint_path.read_bytes()
        report = diagnose_fb_checkpoint(run)
        assert run.read_report(FBCheckpointReport) == report
        assert run.checkpoint_path.read_bytes() == original

    def test_nonuniform_dataset_refused(self, tmp_path, four_rooms_run):
        with pytest.raises(ValueError, match="oracle dataset"):
            diagnose_fb_checkpoint(self.make_run(tmp_path, four_rooms_run, origin="random"))

    def test_dataset_tampering_refused(self, tmp_path, four_rooms_run):
        run = self.make_run(tmp_path, four_rooms_run)
        with run.dataset_path.open("ab") as f:
            f.write(b"tampered")
        with pytest.raises(ValueError, match="fingerprint"):
            diagnose_fb_checkpoint(run)

    def test_bad_coverage_refused_even_with_consistent_hash(self, tmp_path, four_rooms_run):
        transitions = list(FourRooms().all_transitions())
        transitions[1] = transitions[0]
        run = self.make_run(tmp_path, four_rooms_run, transitions=tuple(transitions))
        with pytest.raises(ValueError, match="duplicate or missing"):
            diagnose_fb_checkpoint(run)

    def test_stale_snapshot_is_reported_not_raised(self, tmp_path, four_rooms_run):
        from aeris.agents.fb.checkpoint import load_agent
        from aeris.experiments import record_policy_snapshot

        run = self.make_run(tmp_path, four_rooms_run)
        env, agent = FourRooms(), load_agent(run.checkpoint_path)
        condition = agent.goal_condition(env.observation_for((9, 9)))
        own = record_policy_snapshot(agent, env.positions, env.observation_for, condition=condition)
        run.write_policy_snapshot("final", own)
        assert diagnose_fb_checkpoint(run).final_snapshot_matches_checkpoint is True
        stale = PolicySnapshot(states=own.states, actions=[(a + 1) % 4 for a in own.actions])
        run.write_policy_snapshot("final", stale)
        assert diagnose_fb_checkpoint(run).final_snapshot_matches_checkpoint is False
