"""Infer cluster-star masses from luminosity, temperature, and seismic inputs.

The script cross-matches the fixed-distance luminosity catalogues against
``Kepler_clusters.data`` and the separate Model B/Model E detailed-modelling
density files.  Matches require exact agreement in both KIC and Gaia DR3 ID.

Run from the repository root with::

    python infer_masses.py

The default products are written to ``outputs/mass_inference``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "luminosity-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEFF_SUN_K = 5772.0
NUMAX_SUN_UHZ = 3090.0
OUTLIER_MAD_THRESHOLD = 3.0
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 20261013
EXPECTED_YU18_PHASE = {"RGB": "1", "RC": "2"}
METHOD_COLUMNS = {
    "numax + L + Teff": "Mass_numax_L_Teff_Msun",
    "density Model B + L + Teff": "Mass_density_Model_B_L_Teff_Msun",
    "density Model E + L + Teff": "Mass_density_Model_E_L_Teff_Msun",
}
METHOD_ERROR_COLUMNS = {
    "numax + L + Teff": "Mass_numax_L_Teff_err_Msun",
    "density Model B + L + Teff": "Mass_density_Model_B_L_Teff_err_Msun",
    "density Model E + L + Teff": "Mass_density_Model_E_L_Teff_err_Msun",
}
METHOD_OUTLIER_COLUMNS = {
    "numax + L + Teff": "Mass_numax_outlier_3MAD",
    "density Model B + L + Teff": "Mass_density_Model_B_outlier_3MAD",
    "density Model E + L + Teff": "Mass_density_Model_E_outlier_3MAD",
}
METHOD_STYLES = {
    "numax + L + Teff": ("tab:blue", "o"),
    "density Model B + L + Teff": ("tab:orange", "s"),
    "density Model E + L + Teff": ("tab:green", "^"),
}


@dataclass(frozen=True)
class StateInputs:
    state: str
    luminosity_file: Path
    density_model_b_file: Path
    density_model_e_file: Path


def _normalise_identifier(series: pd.Series) -> pd.Series:
    """Return catalogue identifiers as stripped strings without float suffixes."""
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _prepare_identifiers(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = ["KIC", "GaiaDR3_ID"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing identifier columns: {missing}")

    result = frame.copy()
    for column in required:
        result[column] = _normalise_identifier(result[column])
        if result[column].isna().any() or result[column].eq("").any():
            raise ValueError(f"{label} contains missing {column} identifiers")

    for column in required:
        duplicate = result[column].duplicated(keep=False)
        if duplicate.any():
            values = sorted(result.loc[duplicate, column].unique().tolist())
            raise ValueError(f"{label} has duplicate {column} values: {values}")
    return result


def _validate_pair_mapping(
    base: pd.DataFrame, source: pd.DataFrame, source_label: str
) -> None:
    """Reject a KIC or Gaia match that points to a different counterpart ID."""
    source_by_kic = source.set_index("KIC")["GaiaDR3_ID"]
    source_by_gaia = source.set_index("GaiaDR3_ID")["KIC"]

    kic_overlap = base["KIC"].isin(source_by_kic.index)
    expected_gaia = base.loc[kic_overlap, "KIC"].map(source_by_kic)
    kic_conflict = expected_gaia.ne(base.loc[kic_overlap, "GaiaDR3_ID"])
    if kic_conflict.any():
        bad = base.loc[kic_overlap].loc[kic_conflict, ["KIC", "GaiaDR3_ID"]]
        raise ValueError(
            f"{source_label} has KIC/Gaia pair conflicts: "
            f"{bad.to_dict('records')}"
        )

    gaia_overlap = base["GaiaDR3_ID"].isin(source_by_gaia.index)
    expected_kic = base.loc[gaia_overlap, "GaiaDR3_ID"].map(source_by_gaia)
    gaia_conflict = expected_kic.ne(base.loc[gaia_overlap, "KIC"])
    if gaia_conflict.any():
        bad = base.loc[gaia_overlap].loc[gaia_conflict, ["KIC", "GaiaDR3_ID"]]
        raise ValueError(
            f"{source_label} has Gaia/KIC pair conflicts: "
            f"{bad.to_dict('records')}"
        )


def load_kepler_table(path: Path) -> pd.DataFrame:
    """Read the quoted, whitespace-delimited Kepler cluster compilation."""
    lines = path.read_text().splitlines()
    if not lines or not lines[0].lstrip().startswith("#"):
        raise ValueError(f"{path} does not have the expected commented header")

    header = shlex.split(lines[0].lstrip()[1:].strip())
    rows = [shlex.split(line) for line in lines[1:] if line.strip()]
    bad_lengths = sorted({len(row) for row in rows if len(row) != len(header)})
    if bad_lengths:
        raise ValueError(
            f"{path} has {len(header)} header columns but row lengths {bad_lengths}"
        )

    raw = pd.DataFrame(rows, columns=header)
    selected = raw[
        [
            "KIC_Yu18",
            "GaiaDR3",
            "numax_Yu18",
            "e_numax_Yu18",
            "Phase_Yu18",
            "Kpmag_Yu18",
        ]
    ].rename(
        columns={
            "KIC_Yu18": "KIC",
            "GaiaDR3": "GaiaDR3_ID",
            "numax_Yu18": "numax_uHz",
            "e_numax_Yu18": "numax_err_uHz",
            "Phase_Yu18": "Phase_Yu18",
            "Kpmag_Yu18": "Kp_mag_Yu18",
        }
    )
    selected = _prepare_identifiers(selected, str(path))
    for column in ["numax_uHz", "numax_err_uHz", "Kp_mag_Yu18"]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected["Phase_Yu18"] = selected["Phase_Yu18"].astype("string").str.strip()
    return selected


def load_density_table(path: Path, model: str) -> pd.DataFrame:
    """Read one detailed-modelling density table and preserve all percentiles."""
    raw = pd.read_csv(path, dtype={"KIC": "string", "GaiaDR3_ID": "string"})
    raw = _prepare_identifiers(raw, str(path))
    columns = ["M_R3_ppm_p16", "M_R3_ppm_p50", "M_R3_ppm_p84"]
    missing = [column for column in columns if column not in raw]
    if missing:
        raise ValueError(f"{path} is missing density columns: {missing}")

    selected = raw[["KIC", "GaiaDR3_ID", *columns]].copy()
    rename = {column: column.replace("M_R3", f"M_R3_Model_{model}") for column in columns}
    selected = selected.rename(columns=rename)
    for column in rename.values():
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected


def _merge_source(
    base: pd.DataFrame,
    source: pd.DataFrame,
    source_label: str,
    match_column: str,
) -> pd.DataFrame:
    _validate_pair_mapping(base, source, source_label)
    merged = base.merge(
        source,
        on=["KIC", "GaiaDR3_ID"],
        how="left",
        validate="one_to_one",
        indicator=match_column,
    )
    merged[match_column] = merged[match_column].eq("both")
    return merged


def _valid_positive(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.isfinite(numeric) & numeric.gt(0)


def _robust_outlier_mask(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Flag values farther than three scaled MADs from their sample median."""
    valid = series.dropna()
    mask = pd.Series(False, index=series.index, dtype=bool)
    if valid.empty:
        return mask, np.nan, np.nan
    median = float(valid.median())
    robust_sigma = float(1.4826 * np.median(np.abs(valid - median)))
    if np.isfinite(robust_sigma) and robust_sigma > 0:
        mask.loc[valid.index] = (
            np.abs(valid - median) > OUTLIER_MAD_THRESHOLD * robust_sigma
        )
    return mask, median, robust_sigma


def _weighted_mean(values: pd.Series, errors: pd.Series) -> tuple[float, float]:
    valid = values.notna() & errors.notna() & np.isfinite(errors) & errors.gt(0)
    if not valid.any():
        return np.nan, np.nan
    weights = 1.0 / errors.loc[valid] ** 2
    mean = float(np.sum(weights * values.loc[valid]) / np.sum(weights))
    formal_error = float(np.sqrt(1.0 / np.sum(weights)))
    return mean, formal_error


def _weighted_line(
    x: pd.Series, y: pd.Series, errors: pd.Series
) -> tuple[float, float]:
    valid = (
        x.notna()
        & y.notna()
        & errors.notna()
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(errors)
        & errors.gt(0)
    )
    x_values = x.loc[valid].to_numpy(dtype=float)
    y_values = y.loc[valid].to_numpy(dtype=float)
    weights = 1.0 / errors.loc[valid].to_numpy(dtype=float) ** 2
    weighted_x = np.sum(weights * x_values) / np.sum(weights)
    weighted_y = np.sum(weights * y_values) / np.sum(weights)
    denominator = np.sum(weights * (x_values - weighted_x) ** 2)
    if denominator <= 0:
        return np.nan, np.nan
    slope = np.sum(
        weights * (x_values - weighted_x) * (y_values - weighted_y)
    ) / denominator
    intercept = weighted_y - slope * weighted_x
    return float(slope), float(intercept)


def _mass_from_numax(
    numax_uHz: pd.Series,
    luminosity_lsun: pd.Series,
    teff_k: pd.Series,
) -> pd.Series:
    return (
        (numax_uHz / NUMAX_SUN_UHZ)
        * luminosity_lsun
        * (teff_k / TEFF_SUN_K) ** (-3.5)
    )


def _mass_from_density(
    density_ppm: pd.Series,
    luminosity_lsun: pd.Series,
    teff_k: pd.Series,
) -> pd.Series:
    return (
        density_ppm
        * 1.0e-6
        * luminosity_lsun ** 1.5
        * (teff_k / TEFF_SUN_K) ** (-6.0)
    )


def _add_temperature_sensitivity(
    frame: pd.DataFrame,
    central_column: str,
    method_kind: str,
    input_column: str,
    temperature_shift_k: float,
) -> None:
    central = frame[central_column]
    luminosity = frame["Luminosity_Lsun"]
    teff = frame["teff_used_K"]
    stem = central_column.removesuffix("_Msun")

    for sign, offset in [("minus", -temperature_shift_k), ("plus", temperature_shift_k)]:
        shifted_teff = teff + offset
        if method_kind == "numax":
            shifted = _mass_from_numax(frame[input_column], luminosity, shifted_teff)
        else:
            shifted = _mass_from_density(frame[input_column], luminosity, shifted_teff)
        shifted = shifted.where(central.notna())
        tag = "m" if sign == "minus" else "p"
        shifted_column = f"{stem}_{tag}{temperature_shift_k:g}K_Msun"
        delta_column = f"{stem}_{tag}{temperature_shift_k:g}K_delta_pct"
        frame[shifted_column] = shifted
        frame[delta_column] = 100.0 * (shifted / central - 1.0)


def infer_state_catalogue(
    inputs: StateInputs,
    kepler: pd.DataFrame,
    temperature_shift_k: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    luminosity = pd.read_csv(
        inputs.luminosity_file,
        dtype={"KIC": "string", "GaiaDR3_ID": "string"},
    )
    luminosity = _prepare_identifiers(luminosity, str(inputs.luminosity_file))
    required = [
        "k_mag",
        "Luminosity_Lsun",
        "Luminosity_err_Lsun",
        "teff_used_K",
        "teff_err_used_K",
    ]
    missing = [column for column in required if column not in luminosity]
    if missing:
        raise ValueError(f"{inputs.luminosity_file} is missing columns: {missing}")

    result = luminosity.copy()
    result.insert(2, "Evolutionary_state_catalogue", inputs.state)

    result = _merge_source(
        result,
        kepler,
        "Kepler cluster table",
        "numax_identifier_match_available",
    )
    for model, path in [
        ("B", inputs.density_model_b_file),
        ("E", inputs.density_model_e_file),
    ]:
        density = load_density_table(path, model)
        result = _merge_source(
            result,
            density,
            f"density Model {model}",
            f"density_Model_{model}_identifier_match_available",
        )

    result["numax_available"] = (
        result["numax_identifier_match_available"]
        & _valid_positive(result["numax_uHz"])
        & np.isfinite(result["numax_err_uHz"])
        & result["numax_err_uHz"].ge(0)
    )
    phase_match = pd.Series(pd.NA, index=result.index, dtype="boolean")
    has_phase = result["numax_identifier_match_available"] & result["Phase_Yu18"].notna()
    phase_match.loc[has_phase] = result.loc[has_phase, "Phase_Yu18"].eq(
        EXPECTED_YU18_PHASE[inputs.state]
    )
    result["Phase_Yu18_matches_catalogue"] = phase_match

    luminosity_available = (
        _valid_positive(result["Luminosity_Lsun"])
        & np.isfinite(result["Luminosity_err_Lsun"])
        & result["Luminosity_err_Lsun"].ge(0)
    )
    temperature_available = _valid_positive(result["teff_used_K"])

    numax_mass_mask = result["numax_available"] & luminosity_available & temperature_available
    result["Mass_numax_L_Teff_Msun"] = _mass_from_numax(
        result["numax_uHz"], result["Luminosity_Lsun"], result["teff_used_K"]
    ).where(numax_mass_mask)

    fractional_numax_error = np.sqrt(
        (result["numax_err_uHz"] / result["numax_uHz"]) ** 2
        + (result["Luminosity_err_Lsun"] / result["Luminosity_Lsun"]) ** 2
        + (3.5 * result["teff_err_used_K"] / result["teff_used_K"]) ** 2
    )
    result["Mass_numax_L_Teff_err_Msun"] = (
        result["Mass_numax_L_Teff_Msun"] * fractional_numax_error
    ).where(numax_mass_mask)

    for model in ["B", "E"]:
        density_column = f"M_R3_Model_{model}_ppm_p50"
        availability_column = f"density_Model_{model}_available"
        result[availability_column] = (
            result[f"density_Model_{model}_identifier_match_available"]
            & _valid_positive(result[density_column])
        )
        mass_mask = result[availability_column] & luminosity_available & temperature_available
        mass_column = f"Mass_density_Model_{model}_L_Teff_Msun"
        result[mass_column] = _mass_from_density(
            result[density_column],
            result["Luminosity_Lsun"],
            result["teff_used_K"],
        ).where(mass_mask)
        density_sigma_column = f"M_R3_Model_{model}_ppm_sigma"
        result[density_sigma_column] = 0.5 * (
            result[f"M_R3_Model_{model}_ppm_p84"]
            - result[f"M_R3_Model_{model}_ppm_p16"]
        )
        fractional_density_mass_error = np.sqrt(
            (result[density_sigma_column] / result[density_column]) ** 2
            + (1.5 * result["Luminosity_err_Lsun"] / result["Luminosity_Lsun"]) ** 2
            + (6.0 * result["teff_err_used_K"] / result["teff_used_K"]) ** 2
        )
        result[f"{mass_column.removesuffix('_Msun')}_err_Msun"] = (
            result[mass_column] * fractional_density_mass_error
        ).where(mass_mask)

    _add_temperature_sensitivity(
        result,
        "Mass_numax_L_Teff_Msun",
        "numax",
        "numax_uHz",
        temperature_shift_k,
    )
    for model in ["B", "E"]:
        _add_temperature_sensitivity(
            result,
            f"Mass_density_Model_{model}_L_Teff_Msun",
            "density",
            f"M_R3_Model_{model}_ppm_p50",
            temperature_shift_k,
        )

    for method, mass_column in METHOD_COLUMNS.items():
        outlier_mask, _, _ = _robust_outlier_mask(result[mass_column])
        result[METHOD_OUTLIER_COLUMNS[method]] = outlier_mask
    result["Any_mass_outlier_3MAD"] = result[
        list(METHOD_OUTLIER_COLUMNS.values())
    ].any(axis=1)

    audit: list[dict[str, object]] = []
    source_specs = [
        (
            "Kepler numax",
            "numax_identifier_match_available",
            "numax_available",
            "Mass_numax_L_Teff_Msun",
        ),
        (
            "density Model B",
            "density_Model_B_identifier_match_available",
            "density_Model_B_available",
            "Mass_density_Model_B_L_Teff_Msun",
        ),
        (
            "density Model E",
            "density_Model_E_identifier_match_available",
            "density_Model_E_available",
            "Mass_density_Model_E_L_Teff_Msun",
        ),
    ]
    for source_name, match_column, value_column, mass_column in source_specs:
        missing_ids = result.loc[~result[match_column], "KIC"].tolist()
        audit.append(
            {
                "Evolutionary_state": inputs.state,
                "Source": source_name,
                "Catalogue_rows": len(result),
                "Exact_KIC_Gaia_matches": int(result[match_column].sum()),
                "Usable_source_values": int(result[value_column].sum()),
                "Usable_masses": int(result[mass_column].notna().sum()),
                "Missing_match_KIC": ",".join(missing_ids),
                "Phase_conflict_KIC": (
                    ",".join(
                        result.loc[
                            result["Phase_Yu18_matches_catalogue"].eq(False).fillna(False),
                            "KIC",
                        ].tolist()
                    )
                    if source_name == "Kepler numax"
                    else ""
                ),
            }
        )

    _validate_inferred_catalogue(result, temperature_shift_k)
    return result, audit


def _shifted_mass_column(central_column: str, offset_k: float) -> str:
    if offset_k == 0:
        return central_column
    tag = "m" if offset_k < 0 else "p"
    stem = central_column.removesuffix("_Msun")
    return f"{stem}_{tag}{abs(offset_k):g}K_Msun"


def _shifted_delta_column(central_column: str, offset_k: float) -> str | None:
    if offset_k == 0:
        return None
    tag = "m" if offset_k < 0 else "p"
    stem = central_column.removesuffix("_Msun")
    return f"{stem}_{tag}{abs(offset_k):g}K_delta_pct"


def summarise_masses(
    catalogues: dict[str, pd.DataFrame], temperature_shift_k: float
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state, frame in catalogues.items():
        for method, central_column in METHOD_COLUMNS.items():
            error_column = METHOD_ERROR_COLUMNS[method]
            outlier_column = METHOD_OUTLIER_COLUMNS[method]
            for offset in [-temperature_shift_k, 0.0, temperature_shift_k]:
                mass_column = _shifted_mass_column(central_column, offset)
                delta_column = _shifted_delta_column(central_column, offset)
                valid = frame[mass_column].notna()
                masses = frame.loc[valid, mass_column]
                central_masses = frame.loc[valid, central_column]
                errors = frame.loc[valid, error_column] * masses / central_masses
                outliers = frame.loc[valid, outlier_column].fillna(False)
                kept_masses = masses.loc[~outliers]
                kept_errors = errors.loc[~outliers]
                weighted_mean, weighted_error = _weighted_mean(masses, errors)
                clean_weighted_mean, clean_weighted_error = _weighted_mean(
                    kept_masses, kept_errors
                )
                _, _, robust_sigma = _robust_outlier_mask(masses)
                deltas = (
                    frame.loc[valid, delta_column]
                    if delta_column
                    else pd.Series(0.0, index=masses.index)
                )
                rows.append(
                    {
                        "Evolutionary_state": state,
                        "Method": method,
                        "Teff_offset_K": offset,
                        "N": len(masses),
                        "Mean_mass_Msun": masses.mean(),
                        "Std_mass_Msun": masses.std(ddof=1),
                        "Median_mass_Msun": masses.median(),
                        "Weighted_mean_mass_Msun": weighted_mean,
                        "Weighted_mean_formal_err_Msun": weighted_error,
                        "Robust_sigma_Msun": robust_sigma,
                        "Outlier_rule": (
                            f">{OUTLIER_MAD_THRESHOLD:g} scaled MAD from median"
                        ),
                        "Outlier_KIC": ",".join(
                            frame.loc[masses.index[outliers], "KIC"].tolist()
                        ),
                        "N_without_outliers": len(kept_masses),
                        "Mean_mass_without_outliers_Msun": kept_masses.mean(),
                        "Std_mass_without_outliers_Msun": kept_masses.std(ddof=1),
                        "Median_mass_without_outliers_Msun": kept_masses.median(),
                        "Weighted_mean_without_outliers_Msun": clean_weighted_mean,
                        "Weighted_mean_formal_err_without_outliers_Msun": (
                            clean_weighted_error
                        ),
                        "Mean_fractional_change_pct": deltas.mean(),
                        "Min_fractional_change_pct": deltas.min(),
                        "Max_fractional_change_pct": deltas.max(),
                    }
                )
    return pd.DataFrame(rows)


def _validate_inferred_catalogue(frame: pd.DataFrame, temperature_shift_k: float) -> None:
    for central_column in METHOD_COLUMNS.values():
        central = frame[central_column].dropna()
        if central.empty:
            continue
        if not np.isfinite(central).all() or not central.gt(0).all():
            raise ValueError(f"Invalid masses found in {central_column}")

        lower_teff = frame.loc[
            central.index, _shifted_mass_column(central_column, -temperature_shift_k)
        ]
        higher_teff = frame.loc[
            central.index, _shifted_mass_column(central_column, temperature_shift_k)
        ]
        if not lower_teff.gt(central).all():
            raise ValueError(f"-Teff sensitivity has the wrong sign for {central_column}")
        if not higher_teff.lt(central).all():
            raise ValueError(f"+Teff sensitivity has the wrong sign for {central_column}")


def plot_mass_vs_magnitude(
    catalogues: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True, constrained_layout=True)
    outlier_legend_added = False
    for ax, state in zip(axes, ["RGB", "RC"]):
        frame = catalogues[state]
        for method, mass_column in METHOD_COLUMNS.items():
            color, marker = METHOD_STYLES[method]
            error_column = METHOD_ERROR_COLUMNS[method]
            outlier_column = METHOD_OUTLIER_COLUMNS[method]
            available = frame[
                ["KIC", "k_mag", mass_column, error_column, outlier_column]
            ].dropna(subset=["k_mag", mass_column])
            ax.scatter(
                available["k_mag"],
                available[mass_column],
                label=method,
                color=color,
                marker=marker,
                s=38,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.4,
            )
            outliers = available.loc[available[outlier_column]]
            if len(outliers):
                ax.scatter(
                    outliers["k_mag"],
                    outliers[mass_column],
                    facecolors="none",
                    edgecolors="crimson",
                    s=95,
                    linewidth=1.3,
                    label=("3-MAD candidate" if not outlier_legend_added else None),
                )
                outlier_legend_added = True
                for _, star in outliers.iterrows():
                    ax.annotate(
                        star["KIC"],
                        (star["k_mag"], star[mass_column]),
                        xytext=(4, 5),
                        textcoords="offset points",
                        fontsize=6.5,
                        color="crimson",
                    )
            if state == "RGB":
                clean = available.loc[~available[outlier_column]]
                slope, intercept = _weighted_line(
                    clean["k_mag"], clean[mass_column], clean[error_column]
                )
                if np.isfinite(slope):
                    x_line = np.array([clean["k_mag"].min(), clean["k_mag"].max()])
                    ax.plot(
                        x_line,
                        intercept + slope * x_line,
                        color=color,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.55,
                    )
        ax.invert_xaxis()
        ax.set_title(state)
        ax.set_xlabel(r"2MASS $K_s$ apparent magnitude")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"Mass [$M_\odot$]")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Mass estimates versus apparent magnitude")
    fig.savefig(output_dir / "mass_vs_Ks_magnitude.png", dpi=240)
    fig.savefig(output_dir / "mass_vs_Ks_magnitude.pdf")
    plt.close(fig)


def plot_mass_histograms(
    catalogues: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True, constrained_layout=True)
    for ax, state in zip(axes, ["RGB", "RC"]):
        frame = catalogues[state]
        all_masses = pd.concat(
            [frame[column].dropna() for column in METHOD_COLUMNS.values()],
            ignore_index=True,
        )
        bins = np.histogram_bin_edges(all_masses, bins=max(6, min(10, len(all_masses) // 5)))
        for method, mass_column in METHOD_COLUMNS.items():
            color, _ = METHOD_STYLES[method]
            ax.hist(
                frame[mass_column].dropna(),
                bins=bins,
                histtype="step",
                linewidth=2.0,
                color=color,
                label=method,
            )
        ax.set_title(state)
        ax.set_xlabel(r"Mass [$M_\odot$]")
        ax.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("Number of stars")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("RGB and RC mass distributions")
    fig.savefig(output_dir / "mass_histograms.png", dpi=240)
    fig.savefig(output_dir / "mass_histograms.pdf")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, divider]
    for _, row in frame[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif column in formats:
                cells.append(format(value, formats[column]))
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _bootstrap_mean_difference(
    rgb: pd.Series, rc: pd.Series, rng: np.random.Generator
) -> tuple[float, float]:
    rgb_values = rgb.to_numpy(dtype=float)
    rc_values = rc.to_numpy(dtype=float)
    rgb_indices = rng.integers(
        0, len(rgb_values), size=(BOOTSTRAP_SAMPLES, len(rgb_values))
    )
    rc_indices = rng.integers(
        0, len(rc_values), size=(BOOTSTRAP_SAMPLES, len(rc_values))
    )
    differences = (
        rc_values[rc_indices].mean(axis=1) - rgb_values[rgb_indices].mean(axis=1)
    )
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_weighted_slope(
    frame: pd.DataFrame,
    mass_column: str,
    error_column: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = frame[["k_mag", mass_column, error_column]].dropna().to_numpy(dtype=float)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    sample = values[indices]
    x_values = sample[:, :, 0]
    y_values = sample[:, :, 1]
    weights = 1.0 / sample[:, :, 2] ** 2
    weighted_x = np.sum(weights * x_values, axis=1) / np.sum(weights, axis=1)
    weighted_y = np.sum(weights * y_values, axis=1) / np.sum(weights, axis=1)
    denominator = np.sum(
        weights * (x_values - weighted_x[:, None]) ** 2, axis=1
    )
    slopes = np.sum(
        weights
        * (x_values - weighted_x[:, None])
        * (y_values - weighted_y[:, None]),
        axis=1,
    ) / denominator
    slopes = slopes[np.isfinite(slopes)]
    low, high = np.percentile(slopes, [2.5, 97.5])
    return float(low), float(high)


def build_state_comparison(catalogues: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for method, mass_column in METHOD_COLUMNS.items():
        error_column = METHOD_ERROR_COLUMNS[method]
        outlier_column = METHOD_OUTLIER_COLUMNS[method]
        for sample_name, remove_outliers in [
            ("all", False),
            ("without 3-MAD candidates", True),
        ]:
            samples: dict[str, pd.DataFrame] = {}
            for state in ["RGB", "RC"]:
                sample = catalogues[state][
                    [mass_column, error_column, outlier_column]
                ].dropna(subset=[mass_column])
                if remove_outliers:
                    sample = sample.loc[~sample[outlier_column]]
                samples[state] = sample
            rgb = samples["RGB"]
            rc = samples["RC"]
            ci_low, ci_high = _bootstrap_mean_difference(
                rgb[mass_column], rc[mass_column], rng
            )
            rgb_weighted, _ = _weighted_mean(
                rgb[mass_column], rgb[error_column]
            )
            rc_weighted, _ = _weighted_mean(rc[mass_column], rc[error_column])
            rows.append(
                {
                    "Method": method,
                    "Sample": sample_name,
                    "N_RGB": len(rgb),
                    "N_RC": len(rc),
                    "RC_minus_RGB_mean_Msun": (
                        rc[mass_column].mean() - rgb[mass_column].mean()
                    ),
                    "Mean_difference_bootstrap_p2_5_Msun": ci_low,
                    "Mean_difference_bootstrap_p97_5_Msun": ci_high,
                    "RC_minus_RGB_median_Msun": (
                        rc[mass_column].median() - rgb[mass_column].median()
                    ),
                    "RC_minus_RGB_weighted_mean_Msun": rc_weighted - rgb_weighted,
                }
            )
    return pd.DataFrame(rows)


def build_rgb_magnitude_trends(catalogues: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    frame = catalogues["RGB"]
    rows: list[dict[str, object]] = []
    for method, mass_column in METHOD_COLUMNS.items():
        error_column = METHOD_ERROR_COLUMNS[method]
        outlier_column = METHOD_OUTLIER_COLUMNS[method]
        for sample_name, remove_outliers in [
            ("all", False),
            ("without 3-MAD candidates", True),
        ]:
            sample = frame[
                ["k_mag", mass_column, error_column, outlier_column]
            ].dropna(subset=["k_mag", mass_column, error_column])
            if remove_outliers:
                sample = sample.loc[~sample[outlier_column]]
            slope, _ = _weighted_line(
                sample["k_mag"], sample[mass_column], sample[error_column]
            )
            ci_low, ci_high = _bootstrap_weighted_slope(
                sample, mass_column, error_column, rng
            )
            spearman = sample["k_mag"].rank().corr(sample[mass_column].rank())
            rows.append(
                {
                    "Method": method,
                    "Sample": sample_name,
                    "N": len(sample),
                    "Weighted_slope_Msun_per_mag": slope,
                    "Slope_bootstrap_p2_5_Msun_per_mag": ci_low,
                    "Slope_bootstrap_p97_5_Msun_per_mag": ci_high,
                    "Spearman_rho": spearman,
                }
            )
    return pd.DataFrame(rows)


def build_distance_sensitivity(
    catalogues: dict[str, pd.DataFrame], summary: pd.DataFrame, target_mass: float = 1.15
) -> tuple[pd.DataFrame, float, float, float, float]:
    distance_values = pd.concat(
        [frame["distance_pc"] for frame in catalogues.values()]
    ).dropna()
    distance_error_values = pd.concat(
        [frame["distance_err_pc"] for frame in catalogues.values()]
    ).dropna()
    if distance_values.nunique() != 1 or distance_error_values.nunique() != 1:
        raise ValueError("The fixed-distance catalogues do not share one distance")
    distance_pc = float(distance_values.iloc[0])
    distance_error_pc = float(distance_error_values.iloc[0])
    rgb = summary.loc[
        summary["Evolutionary_state"].eq("RGB")
        & summary["Teff_offset_K"].eq(0)
    ]
    rows: list[dict[str, object]] = []
    for _, row in rgb.iterrows():
        luminosity_exponent = 1.0 if row["Method"].startswith("numax") else 1.5
        distance_exponent = 2.0 * luminosity_exponent
        distance_factor = (target_mass / row["Mean_mass_Msun"]) ** (
            1.0 / distance_exponent
        )
        distance_shift_pc = distance_pc * (distance_factor - 1.0)
        rows.append(
            {
                "Method": row["Method"],
                "Current_RGB_mean_Msun": row["Mean_mass_Msun"],
                "Distance_exponent": distance_exponent,
                "Distance_modulus_shift_to_target_mag": 5.0
                * np.log10(distance_factor),
                "Distance_shift_to_target_pc": distance_shift_pc,
                "Distance_shift_in_sigma": distance_shift_pc / distance_error_pc,
                "RGB_mean_at_plus_1sigma_distance_Msun": row["Mean_mass_Msun"]
                * ((distance_pc + distance_error_pc) / distance_pc)
                ** distance_exponent,
            }
        )
    apparent_modulus = float(catalogues["RGB"]["distance_modulus_input_mag"].dropna().iloc[0])
    apparent_modulus_error = float(
        catalogues["RGB"]["distance_modulus_err_mag"].dropna().iloc[0]
    )
    return (
        pd.DataFrame(rows),
        distance_pc,
        distance_error_pc,
        apparent_modulus,
        apparent_modulus_error,
    )


def write_report(
    output_dir: Path,
    catalogues: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    temperature_shift_k: float,
) -> None:
    method_labels = {
        "numax + L + Teff": r"$\nu_{\max}+L+T_{\rm eff}$",
        "density Model B + L + Teff": r"density B+$L+T_{\rm eff}$",
        "density Model E + L + Teff": r"density E+$L+T_{\rm eff}$",
    }
    baseline = summary.loc[summary["Teff_offset_K"].eq(0)].copy()
    baseline["Method"] = baseline["Method"].replace(method_labels)
    baseline["Mean ± std"] = baseline.apply(
        lambda row: f"{row['Mean_mass_Msun']:.4f} ± {row['Std_mass_Msun']:.4f}",
        axis=1,
    )
    baseline["Weighted mean"] = baseline.apply(
        lambda row: (
            f"{row['Weighted_mean_mass_Msun']:.4f} ± "
            f"{row['Weighted_mean_formal_err_Msun']:.4f}"
        ),
        axis=1,
    )
    baseline_table = _markdown_table(
        baseline,
        [
            "Evolutionary_state",
            "Method",
            "N",
            "Mean ± std",
            "Median_mass_Msun",
            "Weighted mean",
            "Outlier_KIC",
        ],
        {"N": ".0f", "Median_mass_Msun": ".4f"},
    )
    baseline["Clean mean ± std"] = baseline.apply(
        lambda row: (
            f"{row['Mean_mass_without_outliers_Msun']:.4f} ± "
            f"{row['Std_mass_without_outliers_Msun']:.4f}"
        ),
        axis=1,
    )
    baseline["Clean weighted mean"] = baseline.apply(
        lambda row: (
            f"{row['Weighted_mean_without_outliers_Msun']:.4f} ± "
            f"{row['Weighted_mean_formal_err_without_outliers_Msun']:.4f}"
        ),
        axis=1,
    )
    cleaned_table = _markdown_table(
        baseline,
        [
            "Evolutionary_state",
            "Method",
            "N_without_outliers",
            "Clean mean ± std",
            "Median_mass_without_outliers_Msun",
            "Clean weighted mean",
        ],
        {
            "N_without_outliers": ".0f",
            "Median_mass_without_outliers_Msun": ".4f",
        },
    )

    sensitivity = summary.loc[summary["Teff_offset_K"].ne(0)].copy()
    sensitivity["Method"] = sensitivity["Method"].replace(method_labels)
    sensitivity_table = _markdown_table(
        sensitivity,
        [
            "Evolutionary_state",
            "Method",
            "Teff_offset_K",
            "Mean_mass_Msun",
            "Mean_fractional_change_pct",
        ],
        {
            "Teff_offset_K": "+.0f",
            "Mean_mass_Msun": ".4f",
            "Mean_fractional_change_pct": "+.2f",
        },
    )

    state_comparison = build_state_comparison(catalogues)
    state_comparison["Method"] = state_comparison["Method"].replace(method_labels)
    state_comparison["Mean difference 95% CI"] = state_comparison.apply(
        lambda row: (
            f"[{row['Mean_difference_bootstrap_p2_5_Msun']:+.4f}, "
            f"{row['Mean_difference_bootstrap_p97_5_Msun']:+.4f}]"
        ),
        axis=1,
    )
    state_comparison_table = _markdown_table(
        state_comparison,
        [
            "Method",
            "Sample",
            "N_RGB",
            "N_RC",
            "RC_minus_RGB_mean_Msun",
            "Mean difference 95% CI",
            "RC_minus_RGB_median_Msun",
            "RC_minus_RGB_weighted_mean_Msun",
        ],
        {
            "N_RGB": ".0f",
            "N_RC": ".0f",
            "RC_minus_RGB_mean_Msun": "+.4f",
            "RC_minus_RGB_median_Msun": "+.4f",
            "RC_minus_RGB_weighted_mean_Msun": "+.4f",
        },
    )

    trends = build_rgb_magnitude_trends(catalogues)
    trends["Method"] = trends["Method"].replace(method_labels)
    trends["Slope 95% CI"] = trends.apply(
        lambda row: (
            f"[{row['Slope_bootstrap_p2_5_Msun_per_mag']:+.4f}, "
            f"{row['Slope_bootstrap_p97_5_Msun_per_mag']:+.4f}]"
        ),
        axis=1,
    )
    trends_table = _markdown_table(
        trends,
        [
            "Method",
            "Sample",
            "N",
            "Weighted_slope_Msun_per_mag",
            "Slope 95% CI",
            "Spearman_rho",
        ],
        {
            "N": ".0f",
            "Weighted_slope_Msun_per_mag": "+.4f",
            "Spearman_rho": "+.3f",
        },
    )

    (
        distance_sensitivity,
        distance_pc,
        distance_error_pc,
        apparent_modulus,
        apparent_modulus_error,
    ) = build_distance_sensitivity(catalogues, summary)
    distance_sensitivity["Method"] = distance_sensitivity["Method"].replace(
        method_labels
    )
    distance_table = _markdown_table(
        distance_sensitivity,
        [
            "Method",
            "Current_RGB_mean_Msun",
            "Distance_modulus_shift_to_target_mag",
            "Distance_shift_to_target_pc",
            "Distance_shift_in_sigma",
            "RGB_mean_at_plus_1sigma_distance_Msun",
        ],
        {
            "Current_RGB_mean_Msun": ".4f",
            "Distance_modulus_shift_to_target_mag": "+.4f",
            "Distance_shift_to_target_pc": "+.1f",
            "Distance_shift_in_sigma": "+.2f",
            "RGB_mean_at_plus_1sigma_distance_Msun": ".4f",
        },
    )

    audit_table = _markdown_table(
        audit,
        [
            "Evolutionary_state",
            "Source",
            "Catalogue_rows",
            "Exact_KIC_Gaia_matches",
            "Usable_masses",
            "Missing_match_KIC",
            "Phase_conflict_KIC",
        ],
        {
            "Catalogue_rows": ".0f",
            "Exact_KIC_Gaia_matches": ".0f",
            "Usable_masses": ".0f",
        },
    )

    model_notes: list[str] = []
    for state, frame in catalogues.items():
        both = frame.dropna(
            subset=["M_R3_Model_B_ppm_p50", "M_R3_Model_E_ppm_p50"]
        ).copy()
        difference = 100.0 * (
            both["M_R3_Model_E_ppm_p50"] / both["M_R3_Model_B_ppm_p50"] - 1.0
        )
        largest_index = difference.abs().idxmax()
        model_notes.append(
            f"- {state}: median Model E/B density difference "
            f"{difference.median():+.2f}%; largest absolute difference "
            f"{difference.loc[largest_index]:+.2f}% for KIC "
            f"{both.loc[largest_index, 'KIC']}."
        )

    outlier_notes: list[str] = []
    for state, frame in catalogues.items():
        candidate_mask = frame["Any_mass_outlier_3MAD"]
        for _, star in frame.loc[candidate_mask].iterrows():
            flagged = [
                method.replace(" + L + Teff", "")
                for method, flag_column in METHOD_OUTLIER_COLUMNS.items()
                if bool(star[flag_column])
            ]
            mass_values = []
            for method, mass_column in METHOD_COLUMNS.items():
                if pd.notna(star[mass_column]):
                    short_method = method.replace(" + L + Teff", "")
                    mass_values.append(f"{short_method}={star[mass_column]:.3f}")
            outlier_notes.append(
                f"- {state} KIC {star['KIC']}: flagged in {', '.join(flagged)}; "
                f"available masses are {', '.join(mass_values)} $M_\\odot$."
            )

    report = r"""# Mass inference for NGC 6791 RGB and RC stars

## Method

The analysis uses the fixed-distance-modulus luminosity catalogues and requires
exact agreement in both KIC and Gaia DR3 ID for every external match. Catalogue
RGB/RC membership is preserved even when the Yu et al. phase flag differs.

The radius implied by luminosity and temperature is

$$
\frac{R}{R_\odot} = \left(\frac{L}{L_\odot}\right)^{1/2}
\left(\frac{T_{\rm eff}}{T_{{\rm eff},\odot}}\right)^{-2}.
$$

The two mass combinations are

$$
\frac{M_{\nu_{\max}}}{M_\odot} =
\frac{\nu_{\max}}{\nu_{\max,\odot}}
\frac{L}{L_\odot}
\left(\frac{T_{\rm eff}}{T_{{\rm eff},\odot}}\right)^{-7/2}
$$

and

$$
\frac{M_\rho}{M_\odot} =
\frac{\bar\rho}{\bar\rho_\odot}
\left(\frac{L}{L_\odot}\right)^{3/2}
\left(\frac{T_{\rm eff}}{T_{{\rm eff},\odot}}\right)^{-6}.
$$

The adopted constants are $T_{{\rm eff},\odot}=__TEFF_SUN__$ K and
$\nu_{{\max},\odot}=__NUMAX_SUN__$ $\mu$Hz. The detailed-modelling
quantity is interpreted as
$\bar\rho/\bar\rho_\odot=\mathtt{M\_R3\_ppm}/10^6$. Model B and Model E
remain separate throughout.

## Shared distance and the effect of its uncertainty

Yes: every RGB and RC luminosity in these fixed-modulus catalogues uses the same
cluster distance, $__DISTANCE_PC__$ ± $__DISTANCE_ERROR_PC__$ pc. This comes from
the adopted apparent V-band modulus $__APPARENT_MODULUS__$ ±
$__APPARENT_MODULUS_ERROR__$ mag together with the fixed reddening. The distance
uncertainty is therefore a common systematic, not an independent error for each
star.

Because $L\propto d^2$, the two mass estimates scale as
$M_{\nu_{\max}}\propto d^2$ and $M_\rho\propto d^3$. The shifts required to move
the current RGB arithmetic mean to 1.15 $M_\odot$ are:

__DISTANCE_TABLE__

All required shifts are below 0.6 times the adopted one-sigma distance error.
Thus a systematic distance shift well within the uncertainty can move the RGB
means to 1.15 $M_\odot$. At the full +1-sigma distance, the expected means are
about 1.19 $M_\odot$ for the $\nu_{\max}$ method and 1.23 $M_\odot$ for the
density methods.

## Cross-match audit

__AUDIT_TABLE__

One matched RGB star, KIC 2436814, has `Phase_Yu18 = 2` in the Kepler table.
It remains in the RGB sample and is explicitly flagged in the output catalogue.
KIC 2436824 has a detailed-modelling density but no usable luminosity, so it has
no density-derived mass. Missing values are not imputed.

## Mass distributions

The table gives arithmetic means, sample standard deviations (`ddof=1`),
medians, and inverse-variance weighted means. The uncertainty shown beside a
weighted mean is the formal value from the supplied per-star errors. It should
not be interpreted as the uncertainty of the cluster mean because distance,
reddening, and temperature zero-point errors are correlated among stars.
For the density masses, the density standard deviation used for weighting is
approximated as half of the p84-minus-p16 interval and propagated together with
the luminosity and temperature uncertainties.

__BASELINE_TABLE__

### Potential outliers

A candidate is flagged when its mass is farther than three scaled median
absolute deviations from the median for that evolutionary state and method.
This is a diagnostic flag, not an instruction to delete the star.

__OUTLIER_NOTES__

KIC 2437488 and KIC 2437496 are model-specific density discrepancies: each is
flagged in only one detailed-modelling model. KIC 2437805 is different because
all three methods give a high mass, so it could be a genuinely more massive RC
star rather than a bad measurement. It is therefore safest to report results
both with and without these candidates.

After excluding a candidate only from the method or methods that flagged it:

__CLEANED_TABLE__

![Mass estimates versus 2MASS Ks apparent magnitude](mass_vs_Ks_magnitude.png)

The magnitude axis is reversed so that brighter stars appear to the left. Red
circles mark the three-MAD candidates and dashed lines show the RGB weighted
fits after excluding the candidates.

![RGB and RC mass histograms](mass_histograms.png)

## Robustness of the RC–RGB mass difference

The following differences are RC minus RGB. Confidence intervals are
non-parametric 95% bootstrap intervals from resampling stars within each state.

__STATE_COMPARISON_TABLE__

The result is method-dependent. The $\nu_{\max}$ difference is small and its
interval includes zero, so this dataset does not establish an RC–RGB difference
with that method. Both density models give lower RC masses. Model B gives a
clear difference with and without candidate removal. Model E is marginal when
all stars are retained and becomes clearer after candidate removal. Candidate
removal increases rather than erases the density-based difference.

A common distance change multiplies both RGB and RC masses by the same factor
within a given method, so it cannot create or remove the relative difference.
The same is approximately true for a common temperature zero-point shift. The
dominant robustness concern is instead the method and detailed-density model.

## RGB mass versus apparent magnitude

The slopes below use inverse-variance weighted straight-line fits. Their 95%
bootstrap intervals resample stars; Spearman's coefficient provides a separate
monotonic-trend diagnostic.

__TRENDS_TABLE__

Every bootstrap interval includes zero. There is therefore no significant,
outlier-robust RGB mass trend with apparent $K_s$ magnitude. Model E shows the
largest all-star positive slope, but it is driven substantially by the
model-specific low mass of KIC 2437496 and weakens after that candidate is
excluded. The rank correlations are also weak.

## Systematic temperature shift

For this sensitivity test, $T_{\rm eff}$ is changed by
$\pm$__TEMPERATURE_SHIFT__ K while luminosity, $\nu_{\max}$, and mean density
are held fixed. Thus it isolates the explicit temperature term in each mass
relation. It does not include the smaller indirect effect that a temperature
change could have on the bolometric correction and therefore on luminosity.

__SENSITIVITY_TABLE__

As expected from the exponents, density-based masses are more temperature
sensitive than the $\nu_{\max}$ masses: around these temperatures, a +50 K
shift lowers the former by about 6--7% and the latter by about 4%.

## Model B versus Model E densities

__MODEL_NOTES__

These differences are retained rather than averaged, so their effect is visible
both in the catalogues and in the plots.
"""
    report = (
        report.replace("__TEFF_SUN__", f"{TEFF_SUN_K:.0f}")
        .replace("__NUMAX_SUN__", f"{NUMAX_SUN_UHZ:.0f}")
        .replace("__TEMPERATURE_SHIFT__", f"{temperature_shift_k:g}")
        .replace("__DISTANCE_PC__", f"{distance_pc:.1f}")
        .replace("__DISTANCE_ERROR_PC__", f"{distance_error_pc:.1f}")
        .replace("__APPARENT_MODULUS__", f"{apparent_modulus:.2f}")
        .replace(
            "__APPARENT_MODULUS_ERROR__", f"{apparent_modulus_error:.2f}"
        )
        .replace("__DISTANCE_TABLE__", distance_table)
        .replace("__AUDIT_TABLE__", audit_table)
        .replace("__BASELINE_TABLE__", baseline_table)
        .replace("__OUTLIER_NOTES__", "\n".join(outlier_notes))
        .replace("__CLEANED_TABLE__", cleaned_table)
        .replace("__STATE_COMPARISON_TABLE__", state_comparison_table)
        .replace("__TRENDS_TABLE__", trends_table)
        .replace("__SENSITIVITY_TABLE__", sensitivity_table)
        .replace("__MODEL_NOTES__", "\n".join(model_notes))
    )
    (output_dir / "mass_analysis_report.md").write_text(report)


def build_inputs(root: Path) -> list[StateInputs]:
    return [
        StateInputs(
            state="RGB",
            luminosity_file=root
            / "outputs/RGB_all_fixed_distance_modulus/cluster_luminosities_full.csv",
            density_model_b_file=root / "RGB_all_29_stars_with_MR3_Model_B.csv",
            density_model_e_file=root / "RGB_all_29_stars_with_MR3_Model_E.csv",
        ),
        StateInputs(
            state="RC",
            luminosity_file=root
            / "outputs/RC_all_fixed_distance_modulus/cluster_luminosities_full.csv",
            density_model_b_file=root / "RC_all_14_stars_with_MR3_Model_B.csv",
            density_model_e_file=root / "RC_all_14_stars_with_MR3_Model_E.csv",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mass_inference"),
        help="Directory for catalogues, summaries, plots, and report",
    )
    parser.add_argument(
        "--temperature-shift-k",
        type=float,
        default=50.0,
        help="Symmetric Teff systematic to evaluate in kelvin",
    )
    args = parser.parse_args()
    if not np.isfinite(args.temperature_shift_k) or args.temperature_shift_k <= 0:
        raise ValueError("--temperature-shift-k must be a finite positive number")

    root = Path(__file__).resolve().parent
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    kepler = load_kepler_table(root / "Kepler_clusters.data")
    catalogues: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    for inputs in build_inputs(root):
        catalogue, state_audit = infer_state_catalogue(
            inputs, kepler, args.temperature_shift_k
        )
        catalogues[inputs.state] = catalogue
        audit_rows.extend(state_audit)
        catalogue.to_csv(
            output_dir / f"{inputs.state}_mass_catalogue.csv",
            index=False,
            float_format="%.10g",
        )

    summary = summarise_masses(catalogues, args.temperature_shift_k)
    audit = pd.DataFrame(audit_rows)
    summary.to_csv(output_dir / "mass_summary.csv", index=False, float_format="%.10g")
    audit.to_csv(output_dir / "crossmatch_summary.csv", index=False)

    plot_mass_vs_magnitude(catalogues, output_dir)
    plot_mass_histograms(catalogues, output_dir)
    write_report(
        output_dir,
        catalogues,
        summary,
        audit,
        args.temperature_shift_k,
    )

    print(f"Saved mass analysis to {output_dir}")
    print(summary.loc[summary["Teff_offset_K"].eq(0)].to_string(index=False))


if __name__ == "__main__":
    main()
