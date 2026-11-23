# airfoil-designer

Inverse airfoil design tool for fixed-wing drones.

## What it does

You enter mass, span, chord, mission (cruise / range / endurance), minimum thickness and, for cruise, a speed.
The tool:

1. Computes the operating point (Reynolds number, required lift coefficient).
2. Searches an 8-D latent space of airfoil shapes with a fast neural-network surrogate.
3. Checks the result with NeuralFoil against the best classic airfoil in its database (the "gate").
4. Returns the new shape only if NeuralFoil confirms it is better. Otherwise it returns the best classic airfoil.

For range and endurance it also finds the best flight speed.
Output: a plot, the gate verdict, the operating point and a `.dat` coordinate file.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

`optimizer_state.pkl` and `mlp_weights.npz` must be next to `app.py`. numpy is pinned to 2.4.6 because the pickle was made with that version.

## Limitations (honest)

- 2D section analysis only. Rectangular wing, section CL = wing CL, induced drag with Oswald e = 0.85 (range/endurance only).
- Sea-level standard air (rho 1.225 kg/m³, mu 1.81e-5 Pa·s).
- Reynolds number range 5e4–5e5. Cruise points outside it are clamped. Range/endurance only search speeds inside it.
- NeuralFoil trained the surrogate *and* verifies it. The gate is not an independent physics check. "Improvement" means "better per NeuralFoil than the best classic airfoil in this database", not better in the real world.
- The gate does not re-check pitching moment or thickness. The optimizer handles those with penalties.
- No XFOIL/CFD/wind-tunnel validation yet.

## TODO

- Screenshots, example results, deployment notes, license.
