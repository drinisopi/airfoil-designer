"""
app.py - Streamlit front end for airfoil_core.py.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

All the engineering logic lives in airfoil_core.py. This file only:
  1. collects the drone inputs,
  2. calls Core.run(),
  3. shows the result honestly (plot, gate verdict, operating point, .dat download).
"""
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from airfoil_core import RE_MAX, RE_MIN, Core, _grid, drone_to_point, outline

# Files are looked up next to this script, so the app works no matter which
# folder you start `streamlit run` from.
HERE = Path(__file__).parent

# Default drone (the user can change all of these in the sidebar).
DEFAULT_MASS = 2.0     # kg
DEFAULT_SPAN = 2.0     # m
DEFAULT_CHORD = 0.2    # m
DEFAULT_SPEED = 14.0   # m/s (only used for the cruise mission)

# Typical run time on Streamlit Community Cloud (free tier), measured. Shown to the user.
RUN_TIME_TEXT = {"cruise": "about 20 s", "range": "about 2 min", "endurance": "about 2 min"}

# Colors used in both plots (same meaning everywhere).
COLOR_OPT = "#1f6fb4"      # optimized airfoil
COLOR_CLASSIC = "0.55"     # best classic airfoil (gray)

MISSION_HELP = {
    "cruise": "Minimum drag at your chosen speed (also checked at -20 % and +20 % of that speed).",
    "range": "Fly as far as possible: maximise CL/CD of the whole wing. The tool picks the best speed.",
    "endurance": "Stay up as long as possible: maximise CL^1.5/CD of the whole wing. The tool picks the best speed.",
}


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading the model (first time only) ...")
def load_core():
    """Load the PCA basis + MLP weights once and share them between all users/reruns."""
    return Core(state_path=str(HERE / "optimizer_state.pkl"),
                weights_path=str(HERE / "mlp_weights.npz"))


def classic_outline(core, name):
    """Coordinates of a classic airfoil from the database, looked up by its name."""
    names = [str(n) for n in core.names]
    i = names.index(name)                 # first airfoil with this name
    return outline(core.W_real[i])


def fmt(x, spec):
    """Format a number, or show a dash if it is missing (None)."""
    return "-" if x is None else format(x, spec)


def plot_airfoils(result, core):
    """Optimized shape vs best classic. If the gate failed only the classic is drawn,
    because the result then *is* the classic airfoil."""
    fig, ax = plt.subplots(figsize=(9, 2.8))
    classic_xy = classic_outline(core, result["classic_name"])
    if result["passed"]:
        ax.plot(classic_xy[:, 0], classic_xy[:, 1], color=COLOR_CLASSIC, lw=1.5, ls="--",
                label=f"best classic: {result['classic_name']}")
        xy = result["outline"]
        ax.plot(xy[:, 0], xy[:, 1], color=COLOR_OPT, lw=2.2, label="optimized")
    else:
        ax.plot(classic_xy[:, 0], classic_xy[:, 1], color=COLOR_OPT, lw=2.2,
                label=f"best classic: {result['classic_name']} (recommended)")
    ax.set_aspect("equal")                 # same scale on x and y: the shape is not distorted
    ax.set_xlabel("x / c", fontsize=9)
    ax.set_ylabel("y / c", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    # Legend OUTSIDE the axes, just above the top edge, so it never covers the airfoil.
    # Streamlit saves the figure with bbox_inches="tight", so the legend is not cut off.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


def plot_difference(result, core, D, mission):
    """Second plot, only for a verified (gate passed) result:
    left  = shape difference y_optimized - y_classic along the chord (upper and lower surface),
    right = the NeuralFoil numbers the gate compared (no new calculations, only verification rows)."""
    rows = result["verification"]["rows"]
    opt_rows = [r for r in rows if r["who"] == "optimized"]
    cls_rows = [r for r in rows if r["who"] == "classic"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.0), layout="constrained",
                                   gridspec_kw=dict(width_ratios=[1.6, 1]))

    # --- left: shape difference. Both outlines use the same 201 x-points (Selig order:
    # points 0..100 = upper surface from TE to LE, points 100..200 = lower surface LE to TE),
    # so we can subtract the y values point by point.
    xy_o = result["outline"]
    xy_c = classic_outline(core, result["classic_name"])
    dy = 100 * (xy_o[:, 1] - xy_c[:, 1])             # in % of chord
    ax1.axhline(0, color="0.3", lw=0.8)
    ax1.plot(xy_o[:101, 0], dy[:101], color=COLOR_OPT, lw=2, label="upper surface")
    ax1.plot(xy_o[100:, 0], dy[100:], color=COLOR_OPT, lw=2, ls=":", label="lower surface")
    ax1.set_xlabel("x / c", fontsize=9)
    ax1.set_ylabel("y optimized - y classic\n(% of chord)", fontsize=9)
    ax1.set_title("Shape difference", fontsize=10)
    ax1.tick_params(labelsize=8)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, frameon=False)

    # --- right: what the gate measured with NeuralFoil
    if mission == "cruise":
        # Rows are in the order of Core.cruise_points: design speed, -20 %, +20 %.
        # Re-order to slow -> fast so the bars read left to right.
        order = [1, 0, 2]
        labels = [f"{D['V'] * f:.1f} m/s" for f in (0.8, 1.0, 1.2)]
        vals_c = [cls_rows[i]["CD_nf"] for i in order]
        vals_o = [opt_rows[i]["CD_nf"] for i in order]
        vals_c = [None if v is None else v * 1e4 for v in vals_c]   # drag counts (CD x 10^4)
        vals_o = [None if v is None else v * 1e4 for v in vals_o]
        ylabel, title, fmt_spec = "section CD x 10^4 (drag counts)", "Drag per NeuralFoil (lower is better)", ".1f"
    else:
        labels = ["at each airfoil's\nbest speed"]
        vals_c = [cls_rows[0]["metric_nf"]]
        vals_o = [opt_rows[0]["metric_nf"]]
        name = "CL/CD" if mission == "range" else "CL^1.5/CD"
        ylabel, title, fmt_spec = f"wing {name}", f"Wing {name} per NeuralFoil (higher is better)", ".1f"

    # Add the change in % under each group (only where both airfoils have a value).
    labels = [lab if (c is None or o is None) else f"{lab}\n{100 * (o / c - 1):+.1f} %"
              for lab, c, o in zip(labels, vals_c, vals_o)]
    x = np.arange(len(labels))
    bw = 0.38
    for k, (vals, color, lab) in enumerate(((vals_c, COLOR_CLASSIC, "classic"),
                                           (vals_o, COLOR_OPT, "optimized"))):
        for i, v in enumerate(vals):
            if v is None:            # NeuralFoil found no operating point: draw no bar
                continue
            xi = x[i] + (k - 0.5) * bw
            ax2.bar(xi, v, width=bw * 0.95, color=color, label=lab if i == 0 else None)
            ax2.text(xi, v, format(v, fmt_spec), ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x, labels, fontsize=8)
    ax2.set_ylabel(ylabel, fontsize=9)
    ax2.set_title(title, fontsize=10)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.margins(y=0.15)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=8, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    return fig


def reynolds_warnings(D, mission):
    """Warn when the analysis cannot use the real Reynolds number of the drone."""
    msgs = []
    if mission == "cruise":
        # The core evaluates cruise at the design speed and at 0.8x / 1.2x of it.
        # Any Reynolds number outside the training range is silently clamped there.
        for factor, label in ((1.0, "design speed"), (0.8, "-20 % speed"), (1.2, "+20 % speed")):
            Re = D["Re"] * factor
            if not RE_MIN <= Re <= RE_MAX:
                Re_used = float(np.clip(Re, RE_MIN, RE_MAX))
                msgs.append(f"Reynolds number at {label} is {Re:,.0f}, outside {RE_MIN:,.0f}-{RE_MAX:,.0f}. "
                            f"It is clamped to {Re_used:,.0f}, so the results at that point are "
                            f"for a different Reynolds number than your drone actually flies at.")
    return msgs


def cruise_alpha(core, w, D):
    """Angle of attack (per NeuralFoil) at the cruise design point for the final shape.
    run() does not return alpha for cruise, so we evaluate it here with the core's own
    functions (same grid and Re clamping as the verification gate)."""
    AL, _ = _grid("full")
    r = core.eval_point(w, D["Re"], D["CL"], core.sweep_nf, AL)
    return r.get("alpha")


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="Airfoil Designer", layout="wide")
st.title("Airfoil designer for fixed-wing drones")
st.caption("Finds an airfoil for your drone and checks it against the best classic airfoil "
           "in the database. All aerodynamic numbers come from NeuralFoil (2D).")

# --- inputs (in a form, so the slow optimisation only starts when you press the button)
with st.sidebar.form("inputs"):
    st.header("Drone")
    mass = st.number_input("Mass (kg)", min_value=0.1, max_value=50.0, value=DEFAULT_MASS, step=0.1)
    span = st.number_input("Wing span (m)", min_value=0.2, max_value=10.0, value=DEFAULT_SPAN, step=0.1)
    chord = st.number_input("Chord (m)", min_value=0.03, max_value=1.0, value=DEFAULT_CHORD, step=0.01)
    mission = st.radio("Mission", ["cruise", "range", "endurance"], horizontal=True)
    st.caption(MISSION_HELP[mission])
    # Speed is only an input for cruise. For range/endurance the best speed is a RESULT.
    # Note: inside a form, the widgets only update after "Run", so the speed box is
    # always shown but only used for cruise.
    speed = st.number_input("Cruise speed (m/s) - used for cruise only", min_value=3.0,
                            max_value=60.0, value=DEFAULT_SPEED, step=0.5)
    t_min_pct = st.slider("Minimum thickness (% of chord)", min_value=5, max_value=15, value=8)
    submitted = st.form_submit_button("Run design", type="primary")
    st.caption("Run time on the free server: cruise about 20 s, range/endurance about 2 min.")

# For range/endurance the core still needs some speed in the drone dict, but it ignores it
# (it searches over speeds itself). We pass the fixed default so the core's cache stays valid.
V_used = speed if mission == "cruise" else DEFAULT_SPEED
D = drone_to_point(mass, span, chord, V_used)
t_min = t_min_pct / 100

# --- run the optimisation
if submitted:
    core = load_core()
    for msg in reynolds_warnings(D, mission):
        st.warning(msg)

    # run() calls progress(message) at each stage. It does not report a percentage, so
    # we count stages: up to 2 attempts x (optimize + verify) = 4 stages.
    # The 2nd attempt (stages 3-4) only runs if the gate rejects the 1st candidate.
    bar = st.progress(0.0, text="starting ...")
    status = st.empty()                 # one line of text, updated every second
    stage = {"n": 0, "msg": "starting ..."}

    def on_progress(message):
        # Called from the worker thread below. It only stores the text; the Streamlit
        # elements are updated from the main script thread (Streamlit needs that).
        stage["n"] += 1
        stage["msg"] = message

    # Run the slow optimisation in a background thread, so this script can update the
    # elapsed-time text once per second while it waits.
    t_start = time.time()
    pool = ThreadPoolExecutor(max_workers=1)
    job = pool.submit(core.run, D, mission, t_min, effort="fast", progress=on_progress)
    pool.shutdown(wait=False)           # no new jobs; the running one keeps going
    while not job.done():
        n = stage["n"]
        bar.progress(min(n / 4, 0.95), text=f"Stage {n} of max. 4: {stage['msg']}")
        status.caption(f"Running for {time.time() - t_start:.0f} s "
                       f"(a {mission} run usually takes {RUN_TIME_TEXT[mission]} on the free server).")
        wait([job], timeout=1.0)        # sleep up to 1 s, or less if the job finishes

    try:
        result = job.result()           # gives the result, or re-raises the error from run()
    except ValueError as err:          # the core raises readable ValueErrors for infeasible drones
        bar.empty()
        status.empty()
        st.session_state.pop("run", None)
        st.error(f"No design possible: {err}")
        st.stop()
    bar.progress(1.0, text="done")
    status.caption(f"Finished in {time.time() - t_start:.0f} s.")
    # Keep the result across reruns (e.g. clicking the download button reruns the script).
    st.session_state["run"] = dict(result=result, D=D, mission=mission, t_min=t_min,
                                   inputs=dict(mass=mass, span=span, chord=chord, speed=speed))

if "run" not in st.session_state:
    st.info("Set your drone in the sidebar and press **Run design**.")
    st.stop()

# ----------------------------------------------------------------------------- results
core = load_core()
run = st.session_state["run"]
result, D, mission, t_min = run["result"], run["D"], run["mission"], run["t_min"]
ver = result["verification"]
inp = run["inputs"]

st.subheader(f"Result - {mission} mission")
st.caption(f"Inputs: {inp['mass']} kg, span {inp['span']} m, chord {inp['chord']} m, "
           f"min. thickness {t_min:.0%}" + (f", speed {inp['speed']} m/s" if mission == "cruise" else ""))

# --- gate verdict in plain language
metric_word = {"cruise": "lower weighted drag (design speed and +/-20 %)",
               "range": "higher wing CL/CD (range)",
               "endurance": "higher wing CL^1.5/CD (endurance)"}[mission]
if result["passed"]:
    st.success(f"**Verified improvement per NeuralFoil:** {ver['gain_pct']:+.1f} % {metric_word} "
               f"compared with the best classic airfoil in the database ({result['classic_name']}). "
               f"This is a NeuralFoil-vs-NeuralFoil comparison, not a wind-tunnel or flight result.")
else:
    st.warning(f"**No NeuralFoil-verified improvement found.** The recommended airfoil is the best "
               f"classic airfoil for your drone: **{result['classic_name']}**. "
               f"This is by design: the tool only returns a new shape when NeuralFoil confirms it is better.")
    with st.expander("Why the optimized candidate was rejected"):
        for reason in ver["reasons"]:
            st.write(f"- {reason}")

# --- plots
st.pyplot(plot_airfoils(result, core))
if result["passed"]:
    st.markdown("**What actually changed** (the shapes look alike, so here is the difference)")
    st.pyplot(plot_difference(result, core, D, mission))
    st.caption("Left: optimized minus classic surface height. Right: the NeuralFoil numbers "
               "from the gate" + (" at the design speed and +/-20 %" if mission == "cruise" else
                                  ", each airfoil at its own best speed")
               + ". 2D NeuralFoil values, not wind-tunnel data.")
else:
    st.caption("No difference plot: the optimized candidate was rejected, so there is no "
               "verified comparison to show. Its numbers are in 'Gate details' below, labelled as rejected.")

# --- operating point
st.subheader("Operating point (per NeuralFoil, final airfoil)")
cols = st.columns(4)
if mission == "cruise":
    Re_used = float(np.clip(D["Re"], RE_MIN, RE_MAX))
    alpha = cruise_alpha(core, result["w"], D)
    cols[0].metric("Reynolds number", f"{D['Re']:,.0f}",
                   help=None if Re_used == D["Re"] else f"analysed at clamped Re = {Re_used:,.0f}")
    cols[1].metric("Required CL", f"{D['CL']:.3f}")
    cols[2].metric("Angle of attack", fmt(alpha, ".1f") + (" deg" if alpha is not None else ""))
    cols[3].metric("Speed (your input)", f"{D['V']:.1f} m/s")
else:
    # The verification rows hold NeuralFoil's best operating point for each shape.
    who = "optimized" if result["passed"] else "classic"
    row = next(r for r in ver["rows"] if r["who"] == who)
    cols[0].metric("Best speed (result)", fmt(row["V"], ".1f") + (" m/s" if row["V"] else ""))
    cols[1].metric("Reynolds number", fmt(row["Re"], ",.0f"))
    cols[2].metric("CL", fmt(row["CL"], ".3f"))
    cols[3].metric("Angle of attack", fmt(row["alpha"], ".1f") + (" deg" if row["alpha"] is not None else ""))
    st.caption("Best speed is searched only over speeds where the Reynolds number is within "
               f"{RE_MIN:,.0f}-{RE_MAX:,.0f} and a 30 % stall margin is kept.")

st.write(f"Thickness of the final airfoil: **{result['thickness']:.1%}** of chord "
         f"(minimum asked: {t_min:.0%}).")
if result["thickness"] < t_min - 1e-4:
    # The optimizer enforces thickness with a penalty, and the gate does not re-check it.
    st.warning("The final airfoil is thinner than your minimum thickness.")

# --- download
label = result["dat"].splitlines()[0]
st.download_button("Download .dat coordinates", data=result["dat"], file_name=f"{label}.dat",
                   mime="text/plain")

# --- raw gate numbers (for the curious)
with st.expander("Gate details (NeuralFoil numbers for both shapes)"):
    if not result["passed"]:
        st.caption("Rows labelled 'optimized' describe the REJECTED candidate, not the recommended airfoil.")
        if ver["gain_pct"] is not None:
            st.caption(f"Rejected candidate's gain per NeuralFoil: {ver['gain_pct']:+.1f} % "
                       "(not counted: it failed the gate).")
    st.dataframe([{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
                  for r in ver["rows"]], width="stretch")
    st.caption("err_pct = how far the fast surrogate model was from NeuralFoil. conf = NeuralFoil's "
               "own analysis confidence (0-1).")

# --- honest limitations
with st.expander("How this works and limitations"):
    st.markdown(f"""
**How it works**
1. Your mass, span, chord (and speed for cruise) give the Reynolds number and required lift coefficient.
2. A fast neural-network surrogate searches a compressed space of airfoil shapes for the best one.
3. A **gate** re-checks the winner with NeuralFoil and compares it with the best classic airfoil
   from the database. Only if NeuralFoil confirms a gain (and the surrogate agreed with NeuralFoil
   within 5 %, confidence >= 0.7) is the new shape returned. Otherwise you get the classic.

**Limitations**
- **2D analysis only.** Section data; the wing is treated as rectangular with section CL = wing CL.
  For range/endurance, induced drag uses a simple Oswald factor (e = 0.85). No fuselage, tail, propeller wash.
- **Sea-level standard air** (rho = 1.225 kg/m3, mu = 1.81e-5 Pa s).
- **Reynolds number range {RE_MIN:,.0f}-{RE_MAX:,.0f}.** Cruise points outside it are clamped.
- **NeuralFoil is both the teacher of the surrogate and the verifier**, so the gate is *not* an
  independent physics check. A "verified" gain means "better according to NeuralFoil", not
  "better in the real world". Validate with XFOIL/CFD or a wind tunnel before relying on it.
- Improvements are relative to the classic airfoils in this tool's database, not to every real airfoil.
- The gate checks the operating point and model agreement; it does not re-check pitching moment or thickness.
""")
