"""Safe integration checks for Cluster_Luminosity_Pipeline.ipynb.

The validator executes temporary notebook copies with a deterministic mock of the
external bolometric-correction program. It never writes to the real BC directory.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import os
import shutil

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "Cluster_Luminosity_Pipeline.ipynb"
SAMPLE = HERE / "example_cluster_input.csv"

MOCK_BCALL = r'''

# Validation-only replacement: deterministic and sensitive to the MC inputs.
def run_bcall(input_table, label):
    bc = (
        1.50
        + 1.0e-4 * (4500.0 - input_table["teff"].to_numpy())
        + 0.02 * input_table["feh"].to_numpy()
        - 0.01 * (input_table["logg"].to_numpy() - 2.5)
        - 0.3026 * input_table["ebv"].to_numpy()
    )
    return pd.DataFrame({"bc_key": input_table["bc_key"].to_numpy(), "BC_K": bc})
'''


CASES = [
    {
        "name": "fixed_distance_fixed_ebv_fixed_feh",
        "replacements": {
            'DISTANCE_MODE = "fixed_distance_modulus"': 'DISTANCE_MODE = "fixed_distance"',
            "FIXED_FEH = None": "FIXED_FEH = 0.10",
            "FIXED_FEH_ERR = None": "FIXED_FEH_ERR = 0.03",
            "BC_MC_SAMPLES = 200": "BC_MC_SAMPLES = 5",
        },
    },
    {
        "name": "fixed_parallax_fixed_ak",
        "replacements": {
            'DISTANCE_MODE = "fixed_distance_modulus"': 'DISTANCE_MODE = "fixed_parallax"',
            'EXTINCTION_MODE = "fixed_ebv"': 'EXTINCTION_MODE = "fixed_ak"',
            'BC_EXTINCTION_TREATMENT = "inside_bc"': 'BC_EXTINCTION_TREATMENT = "explicit_ak"',
            'BC_ERROR_MODE = "monte_carlo"': 'BC_ERROR_MODE = "fixed"',
        },
    },
    {
        "name": "individual_parallax_constant_zp",
        "replacements": {
            'DISTANCE_MODE = "fixed_distance_modulus"': 'DISTANCE_MODE = "per_star_parallax"',
            'PARALLAX_ZP_MODE = "lindegren"': 'PARALLAX_ZP_MODE = "constant"',
            'BC_ERROR_MODE = "monte_carlo"': 'BC_ERROR_MODE = "none"',
        },
    },
    {
        "name": "cluster_weighted_median_parallax",
        "replacements": {
            'DISTANCE_MODE = "fixed_distance_modulus"':
                'DISTANCE_MODE = "cluster_weighted_median_parallax"',
            "CLUSTER_PARALLAX_MC_SAMPLES = 10_000":
                "CLUSTER_PARALLAX_MC_SAMPLES = 500",
            "BC_MC_SAMPLES = 200": "BC_MC_SAMPLES = 5",
        },
    },
]


def validate_installed_zero_point():
    """Execute through the real Lindegren distance cell with installed packages."""
    notebook = prepare_notebook({
        "name": "real_lindegren",
        "replacements": {
            'DISTANCE_MODE = "fixed_distance_modulus"':
                'DISTANCE_MODE = "per_star_parallax"',
        },
    })
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    with TemporaryDirectory(prefix="luminosity_real_zpt_") as tmp:
        tmp_path = Path(tmp)
        shutil.copy2(SAMPLE, tmp_path / SAMPLE.name)
        previous_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            namespace = {}
            # Configuration, helpers, load/validation, overrides, and distance.
            for cell in code_cells[:5]:
                exec(cell.source, namespace)
        finally:
            os.chdir(previous_cwd)
        data = namespace["dat"]
        assert np.isfinite(data["parallax_zero_point_mas"]).all()
        assert np.isfinite(data["distance_pc"]).all()


def validate_requested_distance_profiles():
    """Run the fixed-DM and combined-parallax distance stages on RGB and RC."""
    profiles = [
        ("RGB_ApparentVDistanceModulus.ipynb", 30, "fixed_distance_modulus"),
        ("RC_ApparentVDistanceModulus.ipynb", 14, "fixed_distance_modulus"),
        ("RGB_WeightedMedianParallax.ipynb", 30, "cluster_weighted_median_parallax"),
        ("RC_WeightedMedianParallax.ipynb", 14, "cluster_weighted_median_parallax"),
    ]
    results = []
    for filename, expected_rows, expected_mode in profiles:
        notebook = nbformat.read(HERE / filename, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        with TemporaryDirectory(prefix="luminosity_distance_profile_") as tmp:
            tmp_path = Path(tmp)
            shutil.copy2(HERE / "RGB_all.csv", tmp_path / "RGB_all.csv")
            shutil.copy2(HERE / "RC_all.csv", tmp_path / "RC_all.csv")
            previous_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                namespace = {}
                for cell in code_cells[:5]:
                    exec(cell.source, namespace)
            finally:
                os.chdir(previous_cwd)
            data = namespace["dat"]
            assert len(data) == expected_rows
            assert namespace["DISTANCE_MODE"] == expected_mode
            assert data["distance_available"].all()
            results.append((filename, namespace["distance_description"]))
    return results


def validate_no_zp_dustmap_pipeline():
    """Execute every stage with no zero point, the real map, and mocked BCs."""
    import matplotlib
    matplotlib.use("Agg")
    case = {
        "name": "no_zp_real_dustmap",
        "replacements": {
            'DISTANCE_MODE = "fixed_distance_modulus"':
                'DISTANCE_MODE = "per_star_parallax"',
            'PARALLAX_ZP_MODE = "lindegren"': 'PARALLAX_ZP_MODE = "none"',
            'EXTINCTION_MODE = "fixed_ebv"': 'EXTINCTION_MODE = "dustmap"',
            "BC_MC_SAMPLES = 200": "BC_MC_SAMPLES = 5",
        },
    }
    notebook = prepare_notebook(case)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    with TemporaryDirectory(prefix="luminosity_no_zp_dustmap_") as tmp:
        tmp_path = Path(tmp)
        shutil.copy2(SAMPLE, tmp_path / SAMPLE.name)
        previous_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            namespace = {}
            for cell in code_cells:
                exec(cell.source, namespace)
        finally:
            os.chdir(previous_cwd)
        compact = pd.read_csv(tmp_path / "outputs" / "cluster_luminosities_HRD.csv")
        full = pd.read_csv(tmp_path / "outputs" / "cluster_luminosities_full.csv")
        assert "extinction_available" in full.columns
        assert 0 < len(compact) <= len(full)
        assert np.isfinite(compact["Luminosity_Lsun"]).all()
        assert (tmp_path / "outputs" / "HR_diagram.png").is_file()


def validate_bc_after_catalogue_reload():
    """Confirm the BC cell recreates `*_used` columns after `dat` is reloaded."""
    case = {
        "name": "bc_after_reload",
        "replacements": {
            'DISTANCE_MODE = "fixed_distance_modulus"': 'DISTANCE_MODE = "fixed_distance"',
            "BC_MC_SAMPLES = 200": "BC_MC_SAMPLES = 5",
        },
    }
    notebook = prepare_notebook(case)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    with TemporaryDirectory(prefix="luminosity_bc_reload_") as tmp:
        tmp_path = Path(tmp)
        shutil.copy2(SAMPLE, tmp_path / SAMPLE.name)
        previous_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            namespace = {}
            # Run config, helpers, and catalogue load; deliberately skip Section 3,
            # then create distance/extinction inputs before the BC stage.
            for cell in code_cells[:3]:
                exec(cell.source, namespace)
            exec(code_cells[4].source, namespace)
            exec(code_cells[5].source, namespace)
            exec(code_cells[6].source, namespace)  # bolometric-correction section
        finally:
            os.chdir(previous_cwd)
        data = namespace["dat"]
        assert data["BC_available"].all()
        assert {"logg_used", "feh_used", "teff_used_K"}.issubset(data.columns)


def prepare_notebook(case, input_name="example_cluster_input.csv"):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    config = notebook.cells[2]
    config.source = config.source.replace(
        'INPUT_FILE = Path("RGB_all.csv")',
        f'INPUT_FILE = Path("{input_name}")',
    )
    config.source = config.source.replace(
        'OUTPUT_DIR = OUTPUT_ROOT / f"{INPUT_FILE.stem}_{DISTANCE_MODE}"',
        'OUTPUT_DIR = Path("outputs")',
    )
    for old, new in case["replacements"].items():
        if old not in config.source:
            raise AssertionError(f"Configuration text not found: {old}")
        config.source = config.source.replace(old, new)

    # The helper-function cell is the first code cell after the configuration.
    notebook.cells[4].source += MOCK_BCALL
    return notebook


def check_outputs(root, case):
    compact = pd.read_csv(root / "outputs" / "cluster_luminosities_HRD.csv")
    full = pd.read_csv(root / "outputs" / "cluster_luminosities_full.csv")
    expected = ["ID", "Luminosity_Lsun", "Luminosity_err_Lsun", "Teff_K", "Teff_err_K"]
    assert compact.columns.tolist() == expected
    assert len(compact) == 5
    assert np.isfinite(compact.iloc[:, 1:].to_numpy()).all()
    assert (compact["Luminosity_Lsun"] > 0).all()
    assert (compact["Luminosity_err_Lsun"] >= 0).all()
    assert (root / "outputs" / "HR_diagram.png").is_file()
    assert (root / "outputs" / "HR_diagram.pdf").is_file()
    assert (root / "outputs" / "04_BC_extinction_comparison.csv").is_file()
    assert (root / "outputs" / "BC_extinction_comparison.png").is_file()

    if case["name"].startswith("fixed_distance"):
        assert np.allclose(full["distance_pc"], 2000.0)
        assert np.allclose(full["feh_used"], 0.10)
    if case["name"].startswith("fixed_parallax"):
        assert np.allclose(full["parallax_corrected_mas"], 0.500)
    if case["name"].startswith("individual_parallax"):
        expected_zp = np.full(len(full), -0.017)
        assert np.allclose(full["parallax_zero_point_mas"], expected_zp)
    if case["name"].startswith("cluster_weighted"):
        assert full["cluster_parallax_mas"].nunique() == 1
        assert (root / "outputs" / "00_cluster_parallax_solution.csv").is_file()
    return compact, full


def execute_case(case):
    notebook = prepare_notebook(case)

    with TemporaryDirectory(prefix=f"luminosity_{case['name']}_") as tmp:
        tmp_path = Path(tmp)
        shutil.copy2(SAMPLE, tmp_path / SAMPLE.name)
        shutil.copy2(HERE / "RGB_all.csv", tmp_path / "RGB_all.csv")
        shutil.copy2(HERE / "RC_all.csv", tmp_path / "RC_all.csv")
        client = NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(tmp_path)}},
        )
        client.execute()
        check_outputs(tmp_path, case)


def save_mock_fixture():
    case = CASES[0]
    destination = HERE / "mock_test"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SAMPLE, destination / "mock_input.csv")
    notebook = prepare_notebook(case, input_name="mock_input.csv")
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(destination)}},
    )
    client.execute()
    check_outputs(destination, case)

    # Friendly top-level copies make the requested mock input/output easy to find.
    shutil.copy2(
        destination / "outputs" / "cluster_luminosities_HRD.csv",
        destination / "mock_output_HRD.csv",
    )
    shutil.copy2(
        destination / "outputs" / "cluster_luminosities_full.csv",
        destination / "mock_output_full.csv",
    )
    shutil.copy2(
        destination / "outputs" / "HR_diagram.png",
        destination / "mock_HR_diagram.png",
    )
    shutil.copy2(
        destination / "outputs" / "HR_diagram.pdf",
        destination / "mock_HR_diagram.pdf",
    )
    shutil.copy2(
        destination / "outputs" / "04_BC_extinction_comparison.csv",
        destination / "mock_BC_extinction_comparison.csv",
    )
    shutil.copy2(
        destination / "outputs" / "BC_extinction_comparison.png",
        destination / "mock_BC_extinction_comparison.png",
    )
    shutil.copy2(
        destination / "outputs" / "BC_extinction_comparison.pdf",
        destination / "mock_BC_extinction_comparison.pdf",
    )
    print(f"Saved mock fixture: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-mock", action="store_true")
    arguments = parser.parse_args()
    if arguments.save_mock:
        save_mock_fixture()
    else:
        validate_installed_zero_point()
        print("PASS: installed zero_point with NumPy array inputs")
        for validation_case in CASES:
            execute_case(validation_case)
            print(f"PASS: {validation_case['name']}")
