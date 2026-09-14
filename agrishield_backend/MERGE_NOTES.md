# AgriShield Backend — Merge Notes (v3, real MRL data added)

This was built and tested directly against your actual `agrishield_backend`
repo (cloned from https://github.com/kumarms97726-sys/agrishield_backend),
plus the real `sih_agrishield` MRL module contents and real FSSAI data you
shared directly.

## Structure

```
agrishield_backend/
├── main.py                 # UPDATED — mounts /diagnose AND /mrl-check
├── diagnose.py              # unchanged (yours)
├── download_model.py         # unchanged (yours)
├── requirements.txt          # unchanged (yours) — already had everything needed
├── model/
│   ├── class_names.json      # unchanged (yours — real weights)
│   └── mobilenetv2_plant.pth # unchanged (yours — real weights, Daksh159/plant-disease-mobilenetv2)
├── mrl/                      # ported from sih_agrishield, made into an importable package
│   ├── __init__.py           # new — empty, makes `mrl` importable
│   ├── mrl_assessment.py     # only change vs. your version: relative imports (`.mrl_lookup`, `.risk_engine`)
│   ├── mrl_lookup.py         # unchanged (yours)
│   └── risk_engine.py        # unchanged (yours)
└── data/
    └── mrl_data.csv          # your real file, FSSAI Version IX (03-02-2026), 12 rows
```

## What was actually run (real model + real data, no stubbing)

- `/diagnose`: real `mobilenetv2_plant.pth` loaded, real forward pass, real
  Grad-CAM heatmap generated (valid JPEG, confirmed). Tested with a synthetic
  placeholder image (not a real leaf photo) — the confidence number from that
  specific run means nothing; test again with a real photo before trusting it.
- `/mrl-check`: ran against your real 12-row FSSAI CSV across a spread of
  cases, confirming all four status tiers plus the fallback:
  - Tomato / Lambda cyhalothrin / 0.075 mg/kg → WARNING (75% of 0.1 mg/kg MRL)
  - Grapes / Fluxapyroxad / 2.9 mg/kg → NEAR_LIMIT (96.7% of 3.0 mg/kg)
  - Rice / Fluxapyroxad / 0.5 mg/kg → SAFE (10% of 5.0 mg/kg)
  - Dry Chilli / Abamectin / 0.6 mg/kg → DANGER (120% of 0.5 mg/kg)
  - Mango / Lambda cyhalothrin / 0.2 mg/kg → NEAR_LIMIT (exactly 100%)
  - Apple / Chlorantraniliprole / 0.1 mg/kg → UNKNOWN (not in the CSV) — confirmed it fails gracefully rather than crashing
- All three routes (`/`, `/diagnose`, `/mrl-check`) tested together in one
  FastAPI app via TestClient.

This spread is genuinely useful for your demo — you can now show every
possible status tier with real data, not just one canned example.

## Two things still open

1. **Decay model still missing.** `assess_crop_safety()` takes
   `predicted_residue` directly — nothing computes that number from a spray
   date + pesticide half-life yet. `/mrl-check` is honest and fully working
   as "given a residue number, judge if it's safe," not yet "given a spray
   date, predict the residue."
2. **FSSAI-only coverage.** All 12 rows are FSSAI (domestic Indian) limits.
   Your pitch's comparison table also claims EU/Gulf export-market rule
   compliance as a differentiator — nothing in this dataset covers that yet.
   Either source EU Reg 396/2005 / Gulf equivalents for the same crop-pesticide
   pairs (add a `destination_market` column since `mrl_lookup.py` currently
   assumes one MRL per crop-pesticide pair, not one per market), or scope the
   demo's MRL claims to domestic compliance only for this round.

## To run

```
pip install -r requirements.txt
uvicorn main:app --reload
```
Run from inside the repo root — `diagnose.py` loads `model/...` as a relative path.
