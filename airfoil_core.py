"""
airfoil_core.py - inverse airfoil design for fixed-wing drones (core logic, no UI).

Pipeline: drone inputs -> operating point -> optimize a shape in an 8-D PCA latent space
using an MLP surrogate -> verify with NeuralFoil (hard gate) -> fall back to the best
classic airfoil if the gate fails.

Dependencies: numpy, scipy, neuralfoil (only for verification). No torch / sklearn needed:
the MLP ensemble is exported once to numpy arrays (see export_numpy_weights).

Files needed next to this module:
    optimizer_state.pkl   PCA basis, real-shape latent coordinates, scalers, bounds
    mlp_weights.npz       MLP ensemble weights (made once from mlp_final_ensemble.pt)

Assumptions (state them in the README): sea-level air (rho 1.225, mu 1.81e-5), rectangular
wing, section CL = wing CL, induced drag with Oswald e = 0.85 (only for range/endurance),
2-D airfoil data from NeuralFoil (the same solver that produced the training labels).
"""
import pickle
import threading
import warnings
from math import comb

import numpy as np
from scipy.optimize import differential_evolution

RHO, MU, G, E_OSW = 1.225, 1.81e-5, 9.81, 0.85
RE_MIN, RE_MAX = 5e4, 5e5            # range the model has seen
CM_MIN, T_MAX = -0.12, 0.22
GATE_CONF, GATE_ERR = 0.7, 5.0       # NeuralFoil confidence >= 0.7, surrogate error <= 5 %
STALL_DESIGN, STALL_OFF = 1.3, 1.05  # CLmax / CL_req margins
CACHE_MAX = 8                         # max. remembered reference results (oldest removed first)
EFFORT = {                            # search effort presets
    "fast": dict(al=21, clg=12, maxiter=40, popsize=10),
    "full": dict(al=41, clg=25, maxiter=100, popsize=15),
}

# ---------------------------------------------------------------- geometry (CST) ---------
_x = (1 - np.cos(np.linspace(0, np.pi, 101))) / 2
_C = np.sqrt(_x) * (1 - _x)
_B = np.stack([comb(5, i) * _x ** i * (1 - _x) ** (5 - i) for i in range(6)], axis=1)


def _surf(w6):
    return _C * (_B @ w6)


def outline(w):
    """12 CST coefficients -> (201, 2) coordinates in Selig order (upper TE -> LE -> lower TE)."""
    yu, yl = _surf(w[:6]), _surf(w[6:])
    return np.column_stack([np.concatenate([_x[::-1], _x[1:]]),
                            np.concatenate([yu[::-1], yl[1:]])])


def thickness(w):
    return float((_surf(w[:6]) - _surf(w[6:])).max())


def to_dat(w, name="OPT"):
    xy = outline(w)
    return name + "\n" + "\n".join(f"{a:.6f} {b:.6f}" for a, b in xy) + "\n"


# ---------------------------------------------------------------- drone -> operating point
def drone_to_point(mass_kg, span_m, chord_m, speed_ms):
    """Level flight at the given speed. Returns the drone dict plus Re and required CL."""
    S = span_m * chord_m
    D = dict(m=mass_kg, S=S, c=chord_m, AR=span_m / chord_m, V=speed_ms)
    D["Re"] = RHO * speed_ms * chord_m / MU
    D["CL"] = 2 * mass_kg * G / (RHO * speed_ms ** 2 * S)
    D["Re_in_range"] = RE_MIN <= D["Re"] <= RE_MAX
    return D


def export_numpy_weights(pt_path, out_path):
    """Run once (needs torch, e.g. in Colab): ensemble .pt -> numpy .npz."""
    import torch
    arrs = {}
    for i, sd in enumerate(torch.load(pt_path, map_location="cpu")):
        vals = [v.detach().cpu().numpy().astype(np.float32) for v in sd.values()]
        assert len(vals) == 8, "unexpected state_dict layout"
        for j in range(4):
            arrs[f"n{i}_W{j}"] = vals[2 * j]        # (out, in)
            arrs[f"n{i}_b{j}"] = vals[2 * j + 1]
    np.savez(out_path, **arrs)


def _grid(effort):
    e = EFFORT[effort]
    return np.linspace(-4, 12, e["al"]), np.linspace(0.2, 1.4, e["clg"])


# ---------------------------------------------------------------- the core ---------------
class Core:
    def __init__(self, state_path="optimizer_state.pkl", weights_path="mlp_weights.npz"):
        st = pickle.load(open(state_path, "rb"))
        self.mean, self.comp = st["pca_mean"], st["pca_components"]      # (12,), (8, 12)
        self.sig = np.sqrt(st["pca_var"])
        self.Z, self.z_lo, self.z_hi = st["Z"], st["z_lo"], st["z_hi"]
        self.mah_max, self.names = float(st["mah_max"]), st["names"]
        sc = st["scalers"]
        self.mu_x, self.sd_x = np.asarray(sc["mu_x"]), np.asarray(sc["sd_x"])
        self.mu_y, self.sd_y = np.asarray(sc["mu_y"]), np.asarray(sc["sd_y"])
        wz = np.load(weights_path)
        n_nets = len({k.split("_")[0] for k in wz.files})
        self.nets = [[(wz[f"n{i}_W{j}"], wz[f"n{i}_b{j}"]) for j in range(4)] for i in range(n_nets)]
        self.W_real = self.z_to_w(self.Z)                                 # (n, 12)
        self.thick_real = np.array([thickness(w) for w in self.W_real])
        # Reference results per drone/mission. One Core is shared by all app visitors, so the
        # cache is kept small (CACHE_MAX entries) and guarded by a lock for concurrent sessions.
        self._cache = {}
        self._cache_lock = threading.Lock()

    # --- latent space
    def z_to_w(self, z):
        return np.atleast_2d(z) @ self.comp + self.mean

    def mah(self, z):
        return float(np.sqrt(((z / self.sig) ** 2).sum()))

    def density(self, z):
        """Mean distance to the 5 nearest real shapes (guard against designing in data gaps)."""
        d = np.sqrt(((self.Z - z) ** 2).sum(1))
        return float(np.partition(d, 4)[:5].mean())

    # --- surrogate (numpy MLP ensemble)
    def predict(self, X):
        """X: (n, 16) = 12 CST, te_up, te_lo, log10(Re), alpha -> (n, 3) = CL, CD, CM."""
        h0 = ((X - self.mu_x) / self.sd_x).astype(np.float32)
        outs = []
        with np.errstate(over="ignore"):
            for net in self.nets:
                h = h0
                for j, (W, b) in enumerate(net):
                    h = h @ W.T + b
                    if j < 3:
                        h = h / (1 + np.exp(-h))                          # SiLU
                outs.append(h)
        out = np.mean(outs, axis=0) * self.sd_y + self.mu_y
        out[:, 1] = np.exp(out[:, 1])
        return out

    def sweep_surr(self, w, Res, AL):
        K, A = len(Res), len(AL)
        X = np.zeros((K * A, 16))
        X[:, :12] = w
        X[:, 14] = np.repeat(np.log10(Res), A)
        X[:, 15] = np.tile(AL, K)
        P = self.predict(X).reshape(K, A, 3)
        return P[..., 0], P[..., 1], P[..., 2], None

    @staticmethod
    def sweep_nf(w, Res, AL):
        import neuralfoil as nf
        xy = outline(w)
        out = [nf.get_aero_from_coordinates(coordinates=xy, alpha=AL, Re=float(R)) for R in Res]
        g = lambda k: np.array([np.asarray(o[k]) for o in out])
        return g("CL"), g("CD"), g("CM"), g("analysis_confidence")

    # --- operating points
    def eval_point(self, w, Re, CL_req, sweep, AL):
        Re = float(np.clip(Re, RE_MIN, RE_MAX))
        CL, CD, CM, CF = sweep(w, np.array([Re]), AL)
        cl = CL[0]
        im = int(cl.argmax())
        out = dict(ok=False, CLmax=float(cl[im]))
        if cl[0] <= CL_req <= cl[im]:
            a = float(np.interp(CL_req, cl[:im + 1], AL[:im + 1]))
            out.update(ok=True, alpha=a, CD=float(np.interp(a, AL, CD[0])),
                       CM=float(np.interp(a, AL, CM[0])),
                       conf=None if CF is None else float(np.interp(a, AL, CF[0])))
        return out

    def best_point(self, w, D, mission, sweep, AL, CLG, stall=STALL_DESIGN):
        """Range/endurance: best CL (hence speed) for this shape. Uses total drag incl. induced."""
        Vs = np.sqrt(2 * D["m"] * G / (RHO * D["S"] * CLG))
        Res = RHO * Vs * D["c"] / MU
        idx = np.where((Res >= RE_MIN) & (Res <= RE_MAX))[0]
        if len(idx) == 0:
            return None
        CL, CD, CM, CF = sweep(w, Res[idx], AL)
        best = None
        for j, k in enumerate(idx):
            cl = CL[j]
            im = int(cl.argmax())
            tgt = CLG[k]
            if not (cl[0] <= tgt <= cl[im]) or tgt * stall > cl[im]:
                continue
            a = float(np.interp(tgt, cl[:im + 1], AL[:im + 1]))
            cdp = float(np.interp(a, AL, CD[j]))
            cdt = cdp + tgt ** 2 / (np.pi * E_OSW * D["AR"])
            met = tgt / cdt if mission == "range" else tgt ** 1.5 / cdt
            if best is None or met > best["metric"]:
                best = dict(metric=float(met), CL=float(tgt), V=float(Vs[k]), Re=float(Res[k]),
                            alpha=a, CD=cdp, CDtot=float(cdt), CM=float(np.interp(a, AL, CM[j])),
                            CLmax=float(cl[im]),
                            conf=None if CF is None else float(np.interp(a, AL, CF[j])))
        return best

    @staticmethod
    def cruise_points(D):
        """(Re, CL_req, weight, stall margin): design speed and +-20 %."""
        return [(RHO * D["V"] * s * D["c"] / MU, D["CL"] / s ** 2, wt, st)
                for s, wt, st in [(1.0, 0.5, STALL_DESIGN), (0.8, 0.25, STALL_OFF), (1.2, 0.25, STALL_OFF)]]

    # --- reference ("best classic") and objective
    def _reference(self, D, mission, t_min, effort):
        key = (mission, round(t_min, 4), effort, tuple(round(float(D[k]), 6) for k in ("m", "S", "c", "AR")),
               round(float(D.get("V", 0)), 6))
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
        # The (slow) computation below runs outside the lock, so other visitors are not blocked.
        AL, CLG = _grid(effort)
        n = len(self.Z)
        if mission == "cruise":
            pts = self.cruise_points(D)
            Re0, CL0 = pts[0][0], pts[0][1]
            cd = np.full(n, np.inf)
            for i in range(n):
                if not (t_min <= self.thick_real[i] <= T_MAX):
                    continue
                r = self.eval_point(self.W_real[i], Re0, CL0, self.sweep_surr, AL)
                if r["ok"] and STALL_DESIGN * CL0 <= r["CLmax"] and r["CM"] >= CM_MIN:
                    cd[i] = r["CD"]
            if not np.isfinite(cd).any():
                raise ValueError("No airfoil can carry this wing loading at this speed with a "
                                 "safe stall margin (required CL too high). Increase speed or wing area.")
            short = [i for i in np.argsort(cd)[:40] if np.isfinite(cd[i])]
            ref_i = short[0]
            ref = [self.eval_point(self.W_real[ref_i], Re, CL, self.sweep_surr, AL)["CD"] for Re, CL, _, _ in pts]
            ref_data = dict(ref=ref, pts=pts, short=short)
        else:
            mets = np.full(n, np.nan)
            for i in range(n):
                if not (t_min <= self.thick_real[i] <= T_MAX):
                    continue
                bp = self.best_point(self.W_real[i], D, mission, self.sweep_surr, AL, CLG)
                if bp and bp["CM"] >= CM_MIN:
                    mets[i] = bp["metric"]
            if np.isnan(mets).all():
                raise ValueError("No feasible airfoil for this drone (Reynolds number out of the "
                                 "5e4-5e5 range or stall margin not reachable).")
            ref_data = dict(ref=float(np.nanmax(mets)), pts=None, short=[int(np.nanargmax(mets))])
        with self._cache_lock:
            self._cache[key] = ref_data
            while len(self._cache) > CACHE_MAX:              # dicts keep insertion order:
                self._cache.pop(next(iter(self._cache)))     # drop the oldest entry
        return ref_data

    def _make_score(self, D, mission, t_min, cap, effort, ref_data):
        AL, CLG = _grid(effort)
        pts, ref = ref_data["pts"], ref_data["ref"]

        def score(z, guard=True):
            pen = 0.0
            if guard:
                pen += max(0.0, self.mah(z) - self.mah_max)
                pen += 5 * max(0.0, self.density(z) - cap)
            w = self.z_to_w(z)[0]
            th = thickness(w)
            pen += 10 * (max(0.0, t_min - th) + max(0.0, th - T_MAX))
            if mission == "cruise":
                tot = 0.0
                for k, (Re, CL, wt, st) in enumerate(pts):
                    r = self.eval_point(w, Re, CL, self.sweep_surr, AL)
                    pen += max(0.0, st * CL - r["CLmax"])
                    if not r["ok"]:
                        pen += 1.0
                        tot += wt * 10
                        continue
                    pen += max(0.0, CM_MIN - r["CM"])
                    tot += wt * r["CD"] / ref[k]
                return tot + 20 * pen
            bp = self.best_point(w, D, mission, self.sweep_surr, AL, CLG)
            if bp is None:
                return 10.0 + 20 * pen
            pen += max(0.0, CM_MIN - bp["CM"])
            return ref / bp["metric"] + 20 * pen
        return score

    # --- optimize + verify
    def optimize(self, D, mission, t_min=0.08, cap=0.10, effort="fast", seed=0):
        rd = self._reference(D, mission, t_min, effort)
        score = self._make_score(D, mission, t_min, cap, effort, rd)
        bi = rd["short"][0]
        if mission == "cruise":   # best classic under the full multi-point objective
            bi = int(min(rd["short"], key=lambda i: score(self.Z[i], guard=False)))
        e = EFFORT[effort]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = differential_evolution(score, list(zip(self.z_lo, self.z_hi)), maxiter=e["maxiter"],
                                         popsize=e["popsize"], seed=seed, tol=1e-8,
                                         init="sobol", polish=False)
        return dict(z=res.x, bi=bi, value=float(res.fun), cap=cap,
                    density=self.density(res.x), thickness=thickness(self.z_to_w(res.x)[0]))

    def verify(self, z, bi, D, mission, effort="full"):
        """NeuralFoil gate at full grid resolution. Returns rows + pass/fail + reasons."""
        AL, CLG = _grid("full")
        wo, wc = self.z_to_w(z)[0], self.W_real[bi]
        rows, reasons = [], []
        if mission == "cruise":
            pts = self.cruise_points(D)
            cds = {}
            for who, w in (("optimized", wo), ("classic", wc)):
                for (Re, CL, wt, st) in pts:
                    s = self.eval_point(w, Re, CL, self.sweep_surr, AL)
                    n = self.eval_point(w, Re, CL, self.sweep_nf, AL)
                    ok = s["ok"] and n["ok"]
                    err = 100 * (s["CD"] / n["CD"] - 1) if ok else None
                    rows.append(dict(who=who, Re=float(np.clip(Re, RE_MIN, RE_MAX)), CL=CL,
                                     CD_nf=n.get("CD"), err_pct=err, CM_nf=n.get("CM"),
                                     conf=n.get("conf"), weight=wt, ok=ok))
            o, c = rows[:3], rows[3:]
            if all(r["ok"] for r in o + c):
                gain = 100 * sum(a["weight"] * (1 - a["CD_nf"] / b["CD_nf"]) for a, b in zip(o, c))
            else:
                gain = None
            for r in o:
                if not r["ok"]:
                    reasons.append("optimized airfoil cannot reach the required CL at all points")
                    break
            if any(r["ok"] and r["conf"] < GATE_CONF for r in o):
                reasons.append(f"NeuralFoil confidence below {GATE_CONF}")
            if any(r["ok"] and abs(r["err_pct"]) > GATE_ERR for r in o):
                reasons.append(f"surrogate differs from NeuralFoil by more than {GATE_ERR}%")
        else:
            res = {}
            for who, w in (("optimized", wo), ("classic", wc)):
                s = self.best_point(w, D, mission, self.sweep_surr, AL, CLG)
                n = self.best_point(w, D, mission, self.sweep_nf, AL, CLG)
                res[who] = (s, n)
                ok = s is not None and n is not None
                rows.append(dict(who=who, V=n and n["V"], Re=n and n["Re"], CL=n and n["CL"],
                                 alpha=n and n["alpha"], metric_nf=n and n["metric"],
                                 err_pct=100 * (s["metric"] / n["metric"] - 1) if ok else None,
                                 CM_nf=n and n["CM"], conf=n and n["conf"], ok=ok))
            o, c = rows
            gain = 100 * (o["metric_nf"] / c["metric_nf"] - 1) if o["ok"] and c["ok"] else None
            if not o["ok"]:
                reasons.append("no feasible operating point for the optimized airfoil")
            else:
                if o["conf"] < GATE_CONF:
                    reasons.append(f"NeuralFoil confidence {o['conf']:.2f} below {GATE_CONF}")
                if abs(o["err_pct"]) > GATE_ERR:
                    reasons.append(f"surrogate differs from NeuralFoil by more than {GATE_ERR}%")
        if gain is None or gain <= 0:
            reasons.append("no NeuralFoil-verified improvement over the best classic airfoil")
        return dict(passed=not reasons, gain_pct=gain, rows=rows, reasons=reasons)

    def run(self, D, mission, t_min=0.08, effort="fast", caps=(0.10, 0.07), progress=None):
        """Optimize -> verify -> (retry with a stricter density cap) -> fall back to the classic."""
        last = None
        for cap in caps:
            if progress:
                progress(f"optimizing (density cap {cap}) ...")
            opt = self.optimize(D, mission, t_min, cap, effort)
            if progress:
                progress("verifying with NeuralFoil ...")
            ver = self.verify(opt["z"], opt["bi"], D, mission)
            last = (opt, ver, cap)
            if ver["passed"]:
                break
        opt, ver, cap = last
        z = opt["z"] if ver["passed"] else self.Z[opt["bi"]]
        w = self.z_to_w(z)[0]
        classic = str(self.names[opt["bi"]])
        label = "OPT_" + mission if ver["passed"] else classic
        return dict(passed=ver["passed"], shape="optimized" if ver["passed"] else "classic",
                    classic_name=classic, cap=cap, verification=ver, w=w, thickness=thickness(w),
                    outline=outline(w), dat=to_dat(w, label), density=opt["density"])
