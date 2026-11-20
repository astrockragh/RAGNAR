# Tier 1 validation of the RAGNAR simulated spectra (2026-10-05)

> **Update, same day:** the issues found below are now fixed, and the checks have been rerun on the fixed code (`_fix1` spectra). See "After the fixes" at the end. The sections in between describe the code *before* the fixes.

These checks need no observed data. Each expected value comes from outside the RAGNAR code path being tested: analytic profile integrals, an independent Boltzmann fit, NIST wavelengths, and MODTRAN read directly with numpy. A pass therefore means more than the code agreeing with itself.

To reproduce, run `python validation/tier1_checks.py` from `RAGNAR_new/` (about 1 min). Raw numbers are in `tier1_raw_output.txt`.

## Summary

| Check | Result |
|---|---|
| 1. Flux bookkeeping | **Pass.** The spectrum integral matches the analytic sum of line profiles to 1e-6 in all three arms. *Known bias:* intensities are 2.45% high, because the line profile isn't normalized. |
| 2. Trot/Tvib round trip | **The pipeline passes:** Trot from the spectra matches Trot from the input line intensities to ≤ 0.35 K, and Tvib to ≤ 16 K. **The emission model has a bug:** rotational energy is counted twice. It is masked by the hot population, so the net error is ≤ 2 K. |
| 3. Wavelength convention | **Pass.** All lines are in vacuum. **One small data error:** Na D1 is 0.84 pm off. |
| 4. Transmission | **Pass.** Peak ratios between PWV 3 mm and 0 mm match MODTRAN read directly to ≤ 0.0005 for isolated lines, including the H₂O band. |
| 5. Line width | **Pass.** Fitted widths match the three LSF scatter factors to ≤ 0.07%. |

## Details

### 1. Flux bookkeeping

**What was checked.** All three arms were tested at 4 temperature combinations each. The integral of each simulated spectrum was compared with the analytic integral of every included line's profile. Each line's window was clipped to ±15σ and to the grid edges, just as the code does.

**Result.** The ratio is 1.000000 in every case, so no line is lost, duplicated or mis-weighted.

**Known bias.** `gauss_exp_scatter` is a unit Gaussian plus an exponential wing that carries an *extra* 2·`r_scat` = 2.5% of flux. Because of that, the spectra integrate to 1.0245–1.0247 × Σ(line intensity × transmission). The truncation at 15σ removes only about 0.03%.

**Effect.** Every intensity, in the spectra and in the line lists, is about 2.5% high. Ratios between lines are unaffected.

**Possible fix.** Divide the profile by (1 + 2·`r_scat`).

### 2. Temperature round trip

**Lines used.** The fit used 14 OH Λ-doublets in the r arm that are isolated, unabsorbed, and have J' ≤ 6.5, from upper levels v' = 3, 5, 6, 7 and 8. Their fluxes were measured from the saved spectra in `sweep_0_0.108_r_20261005.pkl`. Then a Boltzmann fit was done with a common Trot, and Tvib was derived from the level populations.

| Input Trot / Tvib | Trot from spectrum | Trot from model intensities | Tvib from spectrum | Tvib from model intensities |
|---|---|---|---|---|
| 180 K / 8000 K | 180.18 K | 180.23 K | 7934 K | 7927 K |
| 180 K / 12000 K | 181.15 K | 181.20 K | 11846 K | 11842 K |
| 210 K / 8000 K | 208.05 K | 208.38 K | 7960 K | 7947 K |
| 210 K / 12000 K | 209.53 K | 209.87 K | 11899 K | 11883 K |

**The pipeline is faithful.** This covers building the spectra, applying transmission and extracting line fluxes. The spectrum and the model's own line intensities give the same temperatures.

**The emission model has a bug.** In `make_OH_intensities`, the vibrational Boltzmann factor uses each line's *full* upper-state energy (`v = Eupper·hc/kTv`), not the vibrational energy. The rotational energy therefore gets weighted by both exp(−E/kTrot) and exp(−E/kTvib). Fitting the model's line intensities shows this directly:

| Input | Model as is | Hot population off | Hot off, vibrational factor from E_vib only | 1/(1/Trot + 1/Tvib) |
|---|---|---|---|---|
| 180 K / 8000 K | 180.23 K | 176.04 K | 180.00 K | 176.04 K |
| 210 K / 8000 K | 208.38 K | 204.63 K | 210.00 K | 204.63 K |

With the hot population off, the cold component sits at exactly 1/(1/Trot + 1/Tvib), which is 3.4–5.4 K low. The ~1% hot population (`r_hot`) raises the fitted Trot by about 3–4 K, so the net error looks small. That's a coincidence that depends on J range and on v'. The J-dependence of line ratios is still wrong, which matters for climatology and Trot work.

**Possible fix.** Use the minimum `Eupper` of each v' (the vibrational energy) in the vibrational factor. This changes every OH intensity, so the line lists would need regenerating.

**Tvib.** It comes out about 1% low in both the spectrum and the model, so this is the estimator or model rather than the pipeline. I didn't look into it further.

### 3. Wavelength convention

| Line | RAGNAR | NIST air → vacuum | Difference |
|---|---|---|---|
| OI 557.7 | 557.8887 | 557.8888 | −0.08 pm |
| Na D2 | 589.1583 | 589.1583 | −0.02 pm |
| **Na D1** | **589.7567** | **589.7558** | **+0.84 pm** |
| OI 630.0 | 630.2046 | 630.2046 | −0.03 pm |
| OI 636.4 | 636.5536 | 636.5536 | +0.05 pm |
| Hα | 656.4614 | 656.4603 | +1.1 pm (fine-structure blend; within the spread of quoted centroids) |
| Hβ | 486.2721 | 486.2708 | +1.3 pm (fine-structure blend; within the spread of quoted centroids) |

**The data are in vacuum.** Air wavelengths would be 140–180 pm off, consistent with PFS. HITRAN O₂ lines are vacuum by definition. Pgopher OH positions are computed from molecular constants, so they're vacuum too, but checking them against measured sky lines belongs in tier 2.

**Na D1 is 0.84 pm off.** `data/atomic_lines.txt` has 589.756662, but NIST gives 589.75581. That's small next to σ = 91 pm, but larger than the ~0.25 pm `sigma_wav` the line lists report for robust lines.

### 4. Transmission

**What was checked.** OH Λ-doublets in the r arm were compared between PWV 3 mm and PWV 0 mm spectra that share the same temperatures, LSF and OH chemistry draw. The ratio of their spectral peaks was compared with MODTRAN read directly, weighted by each component's Doppler profile.

| Region | Lines | Median t(3 mm)/t(0 mm) | Spectrum − MODTRAN |
|---|---|---|---|
| Clean, 820–890 nm | 9 | 1.000 | ≤ 0.0000 |
| H₂O band, 920–980 nm | 13 | 0.994 | ≤ 0.0005 |
| Whole r arm | 67 | 1.000 | ≤ 0.0062 (from neighbouring lines' wings in the peak) |

The two components of a doublet can see very different transmission inside H₂O bands. For example, (8-4) qQ1(2.5) has 0.81 and 0.62 at 3 mm. RAGNAR handles this correctly, per component.

### 5. Line width

The two most isolated doublets were fitted as two components each, using the RAGNAR profile with σ free:

| LSF scatter | Input width factor | Fitted |
|---|---|---|
| 0 | 0.9730 | 0.9736, 0.9731 |
| 1 | 1.0000 | 1.0007, 1.0001 |
| 2 | 1.0270 | 1.0277, 1.0271 |

Fitting each doublet as a single line instead makes the width look 1.1% too large. That's worth remembering when measuring LSF widths from PFS data.

## Other things noticed along the way (not fixed)

- **H uses the OI red ratio.** In `get_spectra_and_peaks`, H lines are scaled by `I_ratio[4]`, the OI(1D) "red" ratio. The ratio array has no entry for H, so this may be intentional, but it isn't commented.
- **The ±15σ profile cutoff is still a hard step.** It creates the fake peaks seen in the `cent=0` lists. The centroider hides it.
- **Bugs listed earlier, still open:**
  - the v = 8 overpopulation factor is overwritten by an `if`/`if`/`else`;
  - the aerosol bit is always set when transmission < 0.95;
  - `detection_prob` is normalized by max(counts);
  - in `cent=0` runs, the fallback after failed matching retries drops lines.

## Not covered in tier 1 (needs observed spectra, tier 2)

- Whether the pgopher OH positions and the model's relative band intensities match the real sky.
- Whether the line-profile wing parameters (`r_scat` = 0.0125, wing scale 3.4σ) match PFS.
- Whether the model misses lines the sky shows, or predicts lines it doesn't.

## After the fixes (`_fix1`, rerun with `python validation/tier1_checks.py /tigress/cj1223/spectra/sweep_0_0.108_r_fix1_20261005.pkl`)

Raw numbers are in `tier1_raw_output_fix1.txt`. The individual changes are listed in `TIER2_CHECKLIST.md`.

### Check results

| Check | Result |
|---|---|
| 1. Flux bookkeeping | The integral matches the analytic expectation to 1e-6. Integral / Σ(line intensity) = 1.00000 (r arm), 0.99997 (n arm), 0.9998 (b arm, from lines cut off at the grid edges). The 2.45% excess is gone. |
| 2. Temperature round trip | With the hot population switched off, the fit gives back the input Trot exactly (180.00 K and 210.00 K). The spectrum and the model's input intensities agree to ≤ 0.5 K in Trot and ≤ 40 K in Tvib. |
| 3. Wavelengths | Na D1 is now −0.01 pm from NIST. |
| 4. Transmission | Unchanged: ≤ 0.0004 in the H₂O band. |
| 5. Line width | Matches to ≤ 0.01%. |

### Effect on fitted Trot

A single-temperature fit to low-J lines now gives Trot **2.5–10 K above the cold input temperature**. How much depends on the J range: about +2.5 K for J′ ≤ 2.5, rising to about +7.5 K for J′ ≤ 4.5–6.5.

At a fixed J range, the fixed model reads about 4 K warmer than the old one. Two things shift it in the same direction:
- the double counting no longer suppresses the cold population;
- the old code also applied exp(−E_rot/kTvib) to the hot population, which suppressed it.

**This matters for the input grid.** `Trots = [180, 210]` is the *cold* temperature. The apparent Trot of the simulated sky is now several K higher than before. If [180, 210] was chosen to bracket observed Trot values, it may need lowering. I haven't changed it, because how a measured PFS Trot relates to the model depends on the J range used, which is a tier 2 question.

Tvib from the spectra now comes out 3–6% low, in both the spectra and the input intensities. It's an estimator effect: the population correction assumes a single rotational temperature. It doesn't come from the spectrum pipeline.

### Effect on the line lists (`_fix1` vs the earlier `_20261005`)

| Arm, centroider | Lines | Robust | Pairs < 1σ | NaI robust | I ratio, v′ = 8 | I ratio, other lines |
|---|---|---|---|---|---|---|
| b, `cent=0` | 509 → 410 | 291 → 356 | 194 → 1 | 2 / 2 | 1.43 | 0.90 |
| b, `cent=1` | 315 → 314 | 282 → 280 | 3 → 3 | 2 / 2 | 1.44 | 0.91 |
| r, `cent=0` | 630 → 521 | 358 → 440 | 236 → 0 | — | 1.43 | 0.90 |
| r, `cent=1` | 396 → 401 | 353 → 358 | 5 → 3 | — | 1.42 | 0.90 |
| n, `cent=0` | 515 → 408 | 265 → 322 | 167 → 4 | — | 1.44 | 0.89 |
| n, `cent=1` | 286 → 297 | 241 → 242 | 3 → 5 | — | 1.44 | 0.90 |

The intensity ratios are medians of I_new / I_old for robust lines matched within 1 pm between the two versions.
