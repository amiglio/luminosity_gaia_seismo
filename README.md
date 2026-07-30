# Cluster luminosity pipeline

Open `Cluster_Luminosity_Pipeline.ipynb`, edit its first configuration cell, and
choose **Run All**. The notebook replaces the six original APG17 steps with one
ordered workflow and does not modify the original notebooks.

The native inputs are `RGB_all.csv` and `RC_all.csv`. Their 21-column schema is
already mapped in `COL`; no renaming is required.

Four ready-to-run notebooks are provided:

- `RGB_ApparentVDistanceModulus.ipynb`
- `RC_ApparentVDistanceModulus.ipynb`
- `RGB_WeightedMedianParallax.ipynb`
- `RC_WeightedMedianParallax.ipynb`

The apparent-modulus notebooks interpret 13.51 +/- 0.06 as `(m-M)_V`. With
`R_V=3.1` and `E(B-V)=0.14 +/- 0.02`, they derive the true modulus and propagate
both uncertainties into distance. The weighted-parallax notebooks combine and
de-duplicate both RGB and RC catalogues, apply the per-star Lindegren correction,
then compute one inverse-variance weighted-median cluster parallax. Its uncertainty
comes from 10,000 Monte Carlo draws of the measured parallaxes.

`distance_method_summary.csv` records the two resulting cluster distances side
by side. They are 4122.9 +/- 163.9 pc from the apparent V modulus and
4134.7 +/- 59.8 pc from the weighted-median corrected parallax.

## Cluster choices

- Distance: apparent/true distance modulus, combined RGB+RC weighted-median
  Lindegren-corrected parallax, individual Gaia parallaxes, one fixed cluster
  parallax, or one fixed cluster distance.
- Extinction: Bayestar 2019 per star, one fixed cluster E(B-V), or one fixed
  cluster A_Ks.
- Metallicity: individual [Fe/H] values, or one fixed cluster [Fe/H].

Every override is copied to an explicit `*_used` column in the full output. Each
run writes to `outputs/<input-name>_<distance-mode>/`, preventing the RGB, RC,
fixed-modulus, and weighted-parallax results from overwriting each other. The
compact `cluster_luminosities_HRD.csv` has columns:

`ID, Luminosity_Lsun, Luminosity_err_Lsun, Teff_K, Teff_err_K`

Within that run folder, the HR diagram is saved as `HR_diagram.png` and
`HR_diagram.pdf`.

The notebook also saves `04_BC_extinction_comparison.csv` and a
PNG/PDF comparison plot. These compare bolometric magnitudes obtained by
passing E(B-V) directly to `bcall` against corrections computed at E(B-V)=0
with an explicit A_Ks subtraction.

## Saved mock test

`mock_test/` contains a permanent five-star test fixture. Its README records the
fixed cluster settings used. The mock output and HR diagram are synthetic: a
deterministic stand-in was used for the external bolometric-correction executable,
so these files test pipeline wiring and output shape, not astrophysical results.

## External dependencies

The default paths already point to the installed `bcall` code and Bayestar map
found while cleaning the original notebooks. The original APG17 dust-map path no
longer exists; the valid path is under `~/astrosoft/DUST_DIR/green19/`.

The Jupyter kernel needs pandas, NumPy, matplotlib, astropy, and dustmaps.
Individual-parallax mode with the Lindegren correction additionally needs the
package that provides `from zero_point import zpt`. Fixed-distance and
fixed-parallax modes do not import it.

The notebook calls `zero_point` through its vector interface. This avoids the
scalar `np.can_cast` incompatibility in `zero_point` 0.0.1 with NumPy 2.x.

`bcall` writes temporary files inside its own installation directory. Do not run
two copies of the notebook concurrently.

## Scientific conventions made explicit

- `BC_1` is treated as the 2MASS Ks bolometric correction, matching the originals.
- The preferred/default route passes E(B-V) directly to `bcall`, as recommended
  by its authors because extinction coefficients vary with stellar parameters.
- The comparison route calculates BC at E(B-V)=0 and subtracts A_Ks explicitly.
- The luminosity uncertainty includes magnitude, distance, extinction, and BC
  uncertainty. For the reddened-BC route, the E(B-V) interval is propagated by
  rerunning BC at its 16th and 84th percentiles.
- `M_BOL_SUN = 4.75`, matching the zero point adopted by the supplied BC tables.
