# NPP Digital Twin — Homework 2, better-results build

A standalone, offline 3D digital twin of the PCTRAN pressurized-water reactor
for Task E (NPPAD accident diagnosis), serving the diagnosis of the
**TaskE_NPPAD (better-results) Homework 2 package** — the build whose report
aligns the diagnosis windows to the plant's real response onset.

**Live (AWS):** http://npp-digital-twin-hw2-2026.s3-website-us-east-1.amazonaws.com

## What it shows

- The package's 12 held-out **test** runs (one per accident class, median
  severity — the same runs its prediction files score), played on the 3D
  plant with channel traces, playback/scrub, and truth reveal.
- **Diagnosis = the package's 1D CNN**, probabilities averaged over seeds
  0/1/2, taken verbatim from the package's saved outputs
  (`results/probs/*.npz`, cross-checked against its
  `outputs/pred_E_CNN_*_W60_test.csv` files — 48/48 run×setting checks pass).
- The CNN reports **one prediction per evaluated window**
  (W = 30/60/120/300/600 s), so the panel shows **window snapshots**: the
  W snapshot appears at t = t0 + W and stays until the next window completes.
  Nothing is interpolated between windows.

## Settings (toggle in the diagnosis panel)

| Setting | Test macro-F1, CNN (package report, mean of 3 seeds) |
|---|---|
| Onset-aligned t0 = 100 s · realistic (default) | W=30 0.961 · W=60 0.967 · W=120 0.968 · W=300 0.960 · W=600 0.971 |
| Onset-aligned t0 = 100 s · all 96 variables | W=30 0.989 · W=60 0.985 · W=120 0.989 · W=300 0.983 · W=600 0.985 |
| Handout t0 = 20 s · realistic | W=30/60 0.189 (ceiling) · W=120 0.976 · W=300 0.971 · W=600 0.960 |
| Handout t0 = 20 s · all 96 variables | W=30/60 0.189 (ceiling) · W=120 0.992 · W=300 0.988 · W=600 0.990 |

The package's main prediction file (seed-averaged probabilities, onset-aligned,
realistic, W = 60 s) scores macro-F1 **0.972**. With handout windows every
W ≤ 60 s window is one of only four distinct inputs, so no model can beat the
0.189 information ceiling — the package's central finding, reproduced here by
construction.

This twin serves **only what the package contains**: no other models, no
conformal sets, no explainability add-ons.

## Deployment

AWS S3 static website, bucket `npp-digital-twin-hw2-2026` (us-east-1),
deployed by the manual `deploy-npp-twin-hw2.yml` workflow in the
`hydrotwin-site` repo (same pattern as the original twin's deploy). It does
not touch the original twin (repo `npp-digital-twin`, bucket
`npp-digital-twin-2026`) or its Pages site.
