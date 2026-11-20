# Tier 2: RAGNAR against the PFS sky (2026-10-07)

**Data.** Four all-sky visits (Slurm run `u/price/ckj-20260602-after`), with 2,140 good sky fibres each:

| Visits | Night | Moon | Airmass |
|---|---|---|---|
| 129688, 129689 | 2025-09-12 | up, 72% lit | 1.33–1.43 |
| 129806, 129807 | 2025-09-13 | down | 1.33–1.43 |

**Input.** The sky is `(FLUX + SKY)/NORM` from `pfsMerged`, median-stacked per visit and spectrograph (`stack_sky.py`). Visit 129689, SM3 was dropped: its n-arm offset is −9× the continuum, with a wavelength trend.

## What was done

| Script | What it does |
|---|---|
| `stack_sky.py` | Median sky per visit and spectrograph (the main input), and over all spectrographs. |
| `sky_model.py`, `fit_all.py` | RAGNAR forward model per arm, fitted to each stack. Free parameters: Trot, LSF σ (with a linear slope across the arm), a wavelength shift, optionally the wing, plus linear amplitudes per OH v′ (or per band, or per rotational level) and per species, and a Legendre polynomial for the continuum and persistence offsets. Pixels dominated by lines with line-specific H₂O/O₂ absorption are masked. |
| `analyze_fits.py` → `TIER2_fit_summary.txt` | Fitted parameters, vibrational populations, species ratios. |
| `line_level.py [fix1\|fix2]` → `TIER2_line_summary_*.txt` | Per-line observed/model ratios, identification rates, unexplained features. |
| `hot_population.py NIGHT` → `figs/hot_population_night{1,2}.png`, `hot_population_night*.json` | Rotational level populations per v′, with double-Boltzmann fits. |
| `lsf_pfsarm.py ARM [--fix-wing R WF]` → `lsf_*.json` | Empirical LSF on native `pfsArm` pixels. |
| `h2o_check.py` | Line-by-line H₂O transmission against MODTRAN. |

## Results

### Line spread and wing → changed in `run_RAGNAR.py` (lists `_fix2`)

| Arm | `pfsMerged`, wing free: σ / r_scat | `pfsArm` at r_scat = 0.0069: σ intrinsic (incl. pixel) | List σ: old → new | List r_scat |
|---|---|---|---|---|
| b | 0.085 / 0.0115 | no isolated lines | 0.091 → **0.088** | 0.0125 |
| r | 0.1095 / 0.0047 | 0.1095 (0.112) | 0.108 → **0.112** | **0.0069** |
| n | 0.099 / 0.0030 | 0.102 (0.1045) | 0.122 → **0.105** | **0.0069** |

- **The wing:** the `pfsArm` per-line fits can't constrain it, because the local backgrounds absorb it, so `r_scat` comes from the whole-arm `pfsMerged` fits. Following your instruction, it's reduced by 1.8 rather than to the fitted value, so the model wing stays slightly wider than observed.
- **σ:** taken from `pfsArm` with the pixel included, because RAGNAR simulates a continuous spectrum while the pipeline measures pixel-integrated lines.
- **Spread between spectrographs:** about ±2% in σ (r: SM1 0.1125, SM2 0.107).
- **Effect on the n list:** with σ 14% narrower, the centroider list goes from 297 to 343 lines (robust 242 → 274). 178 robust lines are now identified, up from 156.

### Vibrational populations → f₈ set to 1.5 (from 1.6)

- v′ = 3–7 sit within 2–3% of a single Boltzmann, with **Tvib ≈ 9,700 K** (16–84%: 9,600–10,700 K).
- v′ = 8 comes out at 0.88× that line. In the model's terms that's f₈ = 1.41, varying by arm from 1.33 (r) to 1.74 (b). It is set to 1.5.
- v′ = 9 comes out at 0.94×, i.e. f₉ = 1.32, against the 1.4 in the code. Not changed.

### Hot rotational population (not changed in the code, see the figures)

Per v′, cold + hot double Boltzmann. Both nights:

| v′ | Night 2: T_cold | Night 2: r_hot | Night 2: T_hot | Night 1: T_cold | Night 1: r_hot | Night 1: T_hot | RAGNAR now: r_hot, T_hot × Tvib/10⁴ |
|---|---|---|---|---|---|---|---|
| 5 | 186 ± 3 K | 0.043 ± 0.003 | 9,700 ± 4,200 K | 206 ± 3 K | 0.031 ± 0.003 | 9,600 ± 6,500 K | 0.0105, 5,952 K |
| 6 | 186 ± 4 | 0.053 ± 0.003 | 2,800 ± 450 | 205 ± 5 | 0.042 ± 0.004 | 2,200 ± 410 | 0.0117, 4,544 |
| 7 | 187 ± 3 | 0.063 ± 0.003 | 2,150 ± 280 | 202 ± 3 | 0.050 ± 0.003 | 2,060 ± 310 | 0.0123, 2,181 |
| 8 | 209 ± 8 | 0.065 ± 0.006 | 4,300 ± 1,800 | 223 ± 8 | 0.054 ± 0.005 | 5,100 ± 3,300 | 0.0470, 1,741 |
| 9 | 174 ± 10 | 0.16 ± 0.06 | 450 ± 75 | 184 ± 10 | 0.14 ± 0.06 | 470 ± 90 | 0.0491, 1,019 |

- **v′ = 5–7:** the hot fraction is 4–6%, about 4–5× RAGNAR's current values. It's lower on the warmer night 1 by about 20%.
- **T_hot for v′ = 6 and 7** (~2,100–2,800 K) is consistent between nights. For v′ = 7 it matches RAGNAR's value closely. For v′ = 5, T_hot isn't constrained: the hot tail is nearly flat up to 2,700 cm⁻¹.
- **v′ = 9** only reaches E ≈ 1,100 cm⁻¹, so its "hot" component is really a mild curvature. It shouldn't be treated as a hot-population measurement.
- **v′ = 3–4** don't reach high enough rotational energies to say anything.
- **χ²/dof is 40–230:** the stack-to-stack errors are tiny, so the parameter errors are scaled up by √(χ²/dof). The single-Boltzmann alternative is 50–100× worse.

### Line intensities within a band

- **Clean, robust lines** have a median observed/model ratio of 0.94–1.03, with a 16–84% spread of ±7–10%, after the per-v′ and per-species scaling.
- **High-J lines (J′ ≥ 9) are 2–3× the model.** That's the hot population above.
- **Weak high-Δv bands in b** ((6-0), (7-1), (9-2)) are 1.5–2.6× the model. These carry under 1% of the model flux, so they were tied to the strongest OH level in the fit.
- **Band ratios within one upper level** ((7-2)/(7-3), (8-3)/(8-4)) agree with the model to about 5% and 20% respectively.
- **Species against OH** vary between nights, as expected from chemistry:

| Species | Night 1 | Night 2 |
|---|---|---|
| Na | 0.71 | 0.68 |
| OI 557.7 | 0.38 | 0.84 |
| OI 630 | 0.59 | 1.34 |
| O₂ b | 0.57 | 1.14 |

### Which lines can be identified (local peak within 0.5σ, S/N > 5, in ≥ 90% of stacks; `_fix2` centroider lists)

| List intensity (kR) | b | r | n |
|---|---|---|---|
| > 10⁻² | 1.00 | 0.99 | 0.95 |
| 10⁻³–10⁻² | 0.88 | 0.92 | 0.85 |
| 10⁻⁴–10⁻³ | 0.40 | 0.49 | — |
| < 10⁻⁴ | 0.10 | 0.07 | — |

- **No missing sky lines at this threshold:** no residual feature with S/N > 20 and > 5% of the local line signal appears in ≥ 75% of stacks on the moon-down night.
- **Features in only one or two spectrographs** (31 in n, 43 in r, 5 in b) are persistence, ghosts or noise.
- **On the moon-up night** the continuum carries solar absorption structure. One weak feature at 546.24 nm matches the vacuum wavelength of Hg I 546.07 nm, but appears in only one stack.

### Transmission (not resolved)

- **The global fit can't determine PWV:** χ² keeps falling up to 8 mm, which is unphysical.
- **Line-by-line,** observed/intrinsic ratios in the H₂O bands (900–950 and 1100–1160 nm) correlate only 0.3–0.45 with MODTRAN at any PWV or wavelength shift.
- **The worst cases** are lines 10–20 pm from H₂O line cores. Example: (8-4) Q1(2.5) is observed at 0.23 of the model, while MODTRAN gives 0.7–0.8. Small position errors there, in pgopher OH or in MODTRAN/HITRAN H₂O, change the transmission a lot.
- **The fits therefore mask these lines.** Their list intensities and `transmission` columns should be treated as uncertain.

### Wavelength offsets (not acted on)

Observed lines sit −15 pm (b), −24 pm (r) and −35 pm (n) from the RAGNAR positions, consistent across spectrographs. This is presumably the earlier wavelength calibration; we're not touching the subline input wavelengths.

## Open items for you

1. **The hot population:** adopt the fitted r_hot (≈0.04–0.06 for v′ = 5–8) and T_hot (≈2,000–3,000 K for v′ = 6–7) in `make_OH_intensities`? Or wait for more nights, given the night-to-night change in r_hot?
2. **f₉:** keep 1.4, or move to ~1.3?
3. **b-arm σ:** it relies on `pfsMerged` only, because there are no isolated b lines on native pixels. A pipeline LSF or arc-lamp width for b would make it firmer.
