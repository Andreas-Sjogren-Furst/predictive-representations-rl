import pytest


def test_render_alignment_from_recorded_data(tmp_path):
    pytest.importorskip('matplotlib')
    from aeris.environments import FourRooms
    from aeris.visualization.fb_checkpoint import render_fb_action_alignment
    env = FourRooms()
    record = {
        'state_positions': [list(pos) for pos in env.positions],
        'state_action_improvement_agreement': [True] * len(env.positions),
        'goal': [9, 9],
    }
    path = render_fb_action_alignment(record, tmp_path / 'alignment.png')
    assert path.is_file() and path.stat().st_size > 1000
