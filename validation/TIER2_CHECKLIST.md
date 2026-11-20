# Things to check against observed PFS sky spectra

These are the code changes made on 2026-10-05, plus the model assumptions they touch. Each one should be confirmed, or tuned, with real spectra.

- Line lists made with the changes are tagged `_fix1` (`lineLists/sweep_{sig}_{cent}_{arm}_fix1_20261005.*`).
- The lists from before the changes are kept as `..._20261005.*` (no tag).
- The original code is in `.backup_20261005/`.

The tier 1 checks (`TIER1_REPORT.md`, `tier1_checks.py`) only show that the code does what the model says. Whether the *model* matches the sky needs the data.

## Physics changes (they change line intensities)

### 1. v′ = 8 overpopulation factor now applied
**Change.** In `make_OH_intensities`, the `if v0 == 8 / if v0 == 9 / else` chain reset v′ = 8 to f = 1. It's now `elif`, so v′ = 8 gets f = 1.60 and v′ = 9 keeps 1.40.

**Effect.** All v′ = 8 bands are 60% brighter relative to the rest. The total OH intensity is renormalized to 1000 kR, so all other OH lines get slightly fainter.

**Check against data.**
- Measure band-integrated intensities of v′ = 8 against v′ = 7 and v′ = 9, using bands in the same arm where possible: (8-3) vs (7-2)/(9-3) in r, (8-4) vs (7-3)/(9-4) in r.
- Fit f₈ and f₉ freely.
- The 1.6 and 1.4 values are hard-coded with no reference in the code. Find where they came from, or replace them with fitted values.

### 2. Vibrational Boltzmann factor uses only the vibrational energy
**Change.** The factor was exp(−E_upper/kTvib), with E_upper including the rotational energy. That counted rotational energy twice, so the cold population had an effective rotational temperature of 1/(1/Trot + 1/Tvib), 3.4–5.4 K too cold. It now uses the lowest E_upper of each v′.

**Effect.** The cold population is now at the input Trot. The ~1% hot population (`r_hot`, `T_hot` from Noll+ 2020) lifts a single-temperature fit to low-J lines by about 3–4 K. Before the fix the two effects cancelled by coincidence.

**Check against data.**
- Fit Trot from low-J P1-branch lines (J′ ≤ 4.5) in a few bands per arm, and compare with the model.
- Look at residuals against J′ up to high J: that's where the hot population matters. Check whether `r_hot` and `T_hot`, which are scaled by Tvib/10⁴ in the code, reproduce the observed high-J excess.
- **The input grid may need re-centring.** After the fix, a single-temperature fit to the simulated spectra gives Trot 2.5–10 K above the input cold Trot, depending on the J range (see `TIER1_REPORT.md`). That's ~4 K warmer than the old code at a fixed J range. Fit Trot from PFS data with the same J range, then choose the `Trots` input grid (currently [180, 210]) so the *simulated* apparent Trot brackets the *observed* one.

### 3. Line profile normalized and tapered
**Change.** `gauss_exp_scatter` integrated to 1 + 2·`r_scat` (1.025), so every intensity was 2.45% high. It's now normalized to 1. The spectrum builder uses the new `utils.line_profile`, which subtracts the profile's value at ±15σ and renormalizes. Before, the profile stopped abruptly at ±15σ, leaving a small step in the spectrum. That step created fake local maxima in the gaps between lines, such as the 588.072/588.073 pair in the 2024 blue list.

**Effect.**
- Intensities are 2.45% lower everywhere.
- The `cent=0` lists lose their truncation artifacts: pairs closer than 1σ drop from 42 to 1 on 580–640 nm.

**Check against data.**
- Fit the line profile of bright isolated lines in `pfsArm`, out to ±15σ. That tests the wing model (`r_scat` = 0.0125, wing scale = 3.4σ; the defaults in `get_spectra_and_peaks`).
- Fit Λ-doublets as two components. Fitting them as one line makes the width look ~1% too large, because the components are ~0.3σ apart.
- The earlier `eval_linelists/measure_lineprofile_and_width.ipynb` found a skewed Gaussian fits better than a symmetric one. That's not in the model.

### 4. Na D1 wavelength
**Change.** `data/atomic_lines.txt` changed from 589.756662 to 589.755810 nm, the NIST vacuum value (5897.5581 Å).

**Check against data.**
- Measure the D1–D2 separation (the model now has 0.59748 nm).
- Measure the absolute D-line positions against the OH-calibrated wavelength solution.
- Also note: the line-list centroids for Na D sit 2–6 pm blue of the true line positions, because they blend with nearby OH lines. The centroider makes this worse (cent=1: 589.1521 vs 589.1583). See 7.

## Line-list bookkeeping (no change to the spectra)

### 5. Absorption-origin bit mask
**Change.** In `add_absorption_origin`, a line with transmission ≤ 0.95 got the *aerosol* bit forced on, and the total (`combin`) bit was left to a window test. Now `combin` = (transmission ≤ 0.95), as the README describes, and every species bit comes from its own test.

**Check against data.** Lines flagged as attenuated should vary with airmass and PWV, and the H₂O-flagged lines should track PWV. This needs visits spanning a range of airmass and PWV.

### 6. `robust` and `detection_prob`
**Change.**
- `detection_prob` was divided by the largest count of any peak group. Groups that collected several peaks per spectrum had counts above 120, which pushed robust lines down to ~0.43. It's now (number of spectra the line is found in) / (number of simulated spectra), which is never above 1.
- `robust` is now "matched one-to-one in every spectrum".

**Check against data.**
- Completeness: what fraction of robust lines are measured at S/N > 5 in a typical PFS sky frame?
- Calibration: do non-robust lines show up in the data at roughly their `detection_prob`?

### 7. Matching ambiguous lines
**Change.** In `get_common_lines`, a single ambiguous line anywhere in an arm (more than one peak within tolerance in some spectrum) used to make the whole arm fail. After 15 retries, every line whose peak moves between spectra was then dropped, about 270 per arm in the `cent=0` runs, Na D among them. Now only the ambiguous lines are taken out, and they're treated as unstable lines. The `cent=1` runs never hit the failure, so they're unaffected.

**Effect.** The `cent=0` lists now contain Na D, as robust lines, and the strong OH lines that were missing.

**Check against data: which peak definition matches the PFS pipeline?**
- `cent=0` uses the peak of the noiseless spectrum. `cent=1` uses the peak after convolving with the LSF.
- Compare both lists' wavelengths with the centroids the PFS pipeline measures (`arcLines`), especially for blends.
- The answer decides which list to use for calibration.

## Model assumptions not changed, to calibrate with data

### 8. Hydrogen intensity ratio
H lines are scaled by `I_ratio[4]`, the OI(1D) "red" ratio, in `get_spectra_and_peaks`. Hα and Hβ are faint and blended with OH, so check whether that matters.

### 9. OH line positions
The pgopher positions are computed from molecular constants, not measured. Compare them with measured centroids of isolated lines, per band and branch, to look for systematic offsets.

### 10. Relative species intensities and their scatter
Check these against the spread across many PFS visits:
- the 1000 kR OH total (Khomich+ 2008);
- 0.4 kR O₂ b-band and 30 kR O₂ a-band;
- the atomic-line intensities in `atomic_lines.txt`;
- the ranges in `make_intensity_ratios` (X_O ± 10%, X_O2 ± 35%, X_O3 ± 15%, and the log-normal X_red).

The `cent=1` list's `sigma_intensity` should then match the observed variability.

### 11. Continuum
`make_continuum` is 80 R/nm, with a (λ/500)^0.2 slope and a Gaussian bump at 595 nm with 10 nm width. Compare with line-free regions.

### 12. Tvib
Tvib recovered from the simulated spectra comes out ~1% low, in both the spectra and the input intensities. The cause is the estimator or model, not the pipeline. Worth understanding before quoting Tvib from data.

### 13. Transmission model
Lines inside H₂O and O₂ bands depend on MODTRAN at PWV 0 and 3 mm, ZD 30°, background aerosols, Mauna Kea. Compare observed intensity deficits of absorbed lines with MODTRAN at the actual observing conditions.
