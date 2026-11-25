"""
make_examples.py - create example_results.json (the "Show example result" button in the app).

It runs the REAL design code (Core.run, effort="fast", exactly like the app's "Run design"
button) for the default drone and all three missions, and saves the results as plain JSON.

Run it again whenever airfoil_core.py, the model files or the default drone change:
    python make_examples.py
Takes about 1-2 minutes.
"""
import os

# Same thread settings as app.py (must come before numpy is imported). They do not change results.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import datetime
import json
from pathlib import Path

import numpy as np

from airfoil_core import Core, drone_to_point

HERE = Path(__file__).parent
# Keep these equal to DEFAULT_* in app.py and the default of its thickness slider (8 %).
MASS, SPAN, CHORD, SPEED, T_MIN = 2.0, 2.0, 0.2, 14.0, 0.08


def to_plain(x):
    """numpy numbers/arrays -> plain Python numbers/lists, so json can write them."""
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):   # np.float64, np.bool_, np.str_ ...
        return x.item()
    return x


def main():
    core = Core(state_path=str(HERE / "optimizer_state.pkl"), weights_path=str(HERE / "mlp_weights.npz"))
    today = datetime.date.today().isoformat()
    runs = {}
    for mission in ("cruise", "range", "endurance"):
        # The app passes the default speed for range/endurance too (they search the speed themselves).
        D = drone_to_point(MASS, SPAN, CHORD, SPEED)
        result = core.run(D, mission, T_MIN, effort="fast")
        print(f"{mission}: gate {'passed' if result['passed'] else 'failed'}, "
              f"gain {result['verification']['gain_pct']:+.3f} % vs {result['classic_name']}")
        # Same structure the app keeps in st.session_state["run"], plus two labels.
        runs[mission] = dict(result=result, D=D, mission=mission, t_min=T_MIN,
                             inputs=dict(mass=MASS, span=SPAN, chord=CHORD, speed=SPEED),
                             example=True, generated=f"generated {today}, effort 'fast'")
    data = dict(note="Precomputed results for the default drone, made by make_examples.py "
                     "with the real design code. Read by app.py.",
                runs=to_plain(runs))
    out = HERE / "example_results.json"
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
