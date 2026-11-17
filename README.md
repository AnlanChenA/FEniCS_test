# FEniCS_test

Validates the `microstructure_ve` DOLFINx backend against an existing ABAQUS
bilinear elastoplastic RVE result, for microstructure sample
`VF_2_ParVF_1_Agg_2`.

**Material model:** matrix and interphase are bilinear (J2, isotropic
hardening) plastic; the particle is linear elastic. Boundary conditions are
periodic + free-lateral (unconstrained transverse strain) uniaxial x-tension,
matching the original ABAQUS `.inp` deck.

## Files

- `run_msve_comparison.py` — coarse smoke test (see below).
- `full_resolution/run_msve_full.py` — full-resolution, quantitative version.
- `S-Eabaqus.txt` — reference ABAQUS stress-strain curve (13 points).
- `S-Emsve.txt` — DOLFINx result from the coarse smoke test (4 points).
- `test.py`, `rve.inp` — minimal quickstart example, unrelated to the
  comparison above.

Run with the `microstructure-ve` conda environment
(`microstructure_ve` + `dolfinx` + `dolfinx_mpc` + `basix` + `petsc4py`).

## Coarse smoke test

`run_msve_comparison.py` downsamples the 400×400-pixel microstructure image
8× to a 50×50 mesh (RVE physical size unchanged, 1 μm × 1 μm) and drives it
through 4 uniformly-spaced load steps up to 10.5% engineering strain. The
goal was to confirm the DOLFINx backend reproduces ABAQUS in both shape and
magnitude before committing to a full-resolution run.

**Result** — DOLFINx stress at each step vs. ABAQUS's curve interpolated to
the same strain:

| strain  | msve (MPa) | ABAQUS interp (MPa) | rel. error |
|---------|-----------:|---------------------:|-----------:|
| 0.02625 |     139.05 |                133.23 |      4.37% |
| 0.05250 |     176.42 |                175.10 |      0.76% |
| 0.07875 |     195.65 |                193.77 |      0.97% |
| 0.10500 |     213.60 |               208.13* |      2.62%*|

\* ABAQUS's recorded curve only extends to ε ≈ 0.0998; this point is
extrapolated beyond ABAQUS's data range and should be treated as indicative
only.

Agreement is within ~1-4% across the covered range, consistent with the
~3-4% systematic gap expected between DOLFINx's full-integration CPE4
elements and ABAQUS's reduced-integration CPE4R elements. The coarse mesh is
enough to confirm the model is wired correctly; it is not a quantitative
validation.

## Full-resolution run — known runtime issue

`full_resolution/run_msve_full.py` uses the true 400×400 mesh (~321,602
DOF) and solves directly at ABAQUS's 13 recorded strain points instead of
interpolating. It has **not yet completed a run**; the input data files
(`*.mat`, `uni_level.txt`) still need to be copied in alongside the script,
and the expected wall-clock time is likely underestimated.

Why it's slow:

- The free-lateral boundary condition (transverse strain not prescribed)
  requires an inner root-find per load step: up to **20 sub-increments ×
  25 root-find iterations × 50 Newton iterations**, each needing a fresh LU
  factorization of the full complex system.
- At 321,602 DOF, the package's own calibrated cost model predicts a
  **single** LU factor+solve at ~73 s (single core). Even a few thousand LU
  solves across the 13 steps — well short of the theoretical worst case —
  puts total runtime in the range of **tens of hours to several days**,
  not the "tens of minutes to a few hours" originally estimated.
- No GPU or iterative-solver shortcut is available: the backend explicitly
  uses PETSc's native serial LU (benchmarked ~25× faster than MUMPS here),
  and GMRES+ILU — the only iterative option for this complex-symmetric
  system (no AMG support for complex scalars) — is documented to diverge
  above ~120k DOF, well below this problem's size. There is no GPU code
  path in `microstructure_ve` at all.

**Before committing to a full run:** calibrate actual per-step iteration
counts on a reduced strain-point list (e.g. 1-2 points) to get a real
wall-clock estimate, rather than running all 13 points blind.
