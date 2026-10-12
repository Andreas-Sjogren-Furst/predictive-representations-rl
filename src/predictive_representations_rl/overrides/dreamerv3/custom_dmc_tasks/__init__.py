"""onestep-fb's ExORL task definitions under the name DreamerV3 imports (`custom_dmc_tasks`).

The two repos ship different `custom_dmc_tasks`: the maze and dynamics match, but Offline_vs_Online_in_MBRL's
point_mass_maze rewards `1 - distance` everywhere while onestep-fb's (ExORL's) rewards only near the goal. The
dreamerv3 adapter puts this directory first on PYTHONPATH so Dreamer trains and is evaluated on exactly the tasks
FB's data, zero-shot evaluation and the probe rewards use, without changing either repo.

Runs inside the Dreamer runtime; it must not import predictive_representations_rl.
"""

import sys
from pathlib import Path

_FB_REPO = Path(__file__).resolve().parents[5] / "third_party" / "onestep-fb"
if str(_FB_REPO) not in sys.path:
    sys.path.append(str(_FB_REPO))  # appended, so `envs` cannot shadow anything Dreamer imports

from envs.custom_dmc_tasks import *  # noqa: E402,F401,F403
from envs.custom_dmc_tasks import make, make_jaco  # noqa: E402,F401
