"""Small synthetic checks; no experimental images are needed."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

os.environ["MPLBACKEND"] = "Agg"

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tifffile

from get_angle_spacing import AngleSpacingCalculator
from si_recon_2D_h import si2D
import sim_recon_2d as runner


def test_group_frames_preserves_angle_phase_order():
    frames = np.arange(18 * 16 * 16).reshape(18, 16, 16)
    grouped = runner.group_frames(frames)
    assert grouped.shape == (2, 9, 16, 16)
    np.testing.assert_array_equal(grouped[1, 0], frames[9])


@pytest.mark.parametrize("shape", [(8, 16, 16), (9, 16, 12), (9, 15, 15), (0, 16, 16)])
def test_rejects_invalid_frame_layout(shape):
    with pytest.raises(ValueError):
        runner.group_frames(np.ones(shape))


def test_read_hdf5_orders_numeric_datasets(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as handle:
        for index in range(12):
            handle[str(index)] = np.full((16, 16), index)
        handle["metadata"] = np.array([999])
    images = runner.read_img(path)
    np.testing.assert_array_equal(images[:, 0, 0], np.arange(12))


def test_unsupported_file_type(tmp_path):
    with pytest.raises(ValueError, match="format"):
        runner.read_img(tmp_path / "input.png")


def test_known_peak_geometry_and_dc_rejection():
    selector = AngleSpacingCalculator(np.ones((128, 128)), dx=0.135)
    angle, spacing = selector.get_ang_spacing_given_brightest_pixel((40, 64))
    assert angle == pytest.approx(np.pi / 2)
    assert spacing == pytest.approx(0.72)
    with pytest.raises(ValueError):
        selector.get_ang_spacing_given_brightest_pixel((64, 64))
    plt.close("all")


def test_peak_selection_handles_reverse_drag():
    spectrum = np.zeros((128, 128))
    spectrum[64, 40] = 10
    selector = AngleSpacingCalculator(spectrum, dx=0.135)
    selector.start_x, selector.start_y = 45, 70
    selector.process_selection(35, 60)
    assert selector.get_brightest_pixel() == (40, 64)
    assert selector.get_ang_spacing()[1] == pytest.approx(0.72)
    plt.close("all")


def test_core_rejects_invalid_images():
    with pytest.raises(ValueError):
        si2D(None)


def test_reconstruction_constant_field_and_no_implicit_figures():
    model = si2D(np.full((9, 128, 128), 5.0))
    model.mu = 0.2
    plt.close("all")
    model.recon([0.0, 2 * np.pi / 3, 4 * np.pi / 3], [0.7] * 3, [1.8] * 3, [0.0] * 3)
    assert model.finalimage.shape == (256, 256)
    assert np.isfinite(model.finalimage).all()
    # A spatially uniform input must remain uniform after linear filtering.
    assert abs(model.finalimage).mean() > 0
    np.testing.assert_allclose(abs(model.finalimage), abs(model.finalimage).mean(), rtol=1e-5)
    assert plt.get_fignums() == []


def test_cli_help_has_no_data_dependency():
    result = subprocess.run([sys.executable, "sim_recon_2d.py", "--help"],
                            cwd=Path(runner.__file__).parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout
    assert "--angles" in result.stdout


def test_phase_separation_recovers_known_harmonics():
    theta = np.linspace(-np.pi, np.pi, 128)[None, :]
    frames = np.stack([np.broadcast_to(7 + 2 * np.cos(theta + phase), (128, 128))
                       for phase in [0, 2 * np.pi / 3, 4 * np.pi / 3]])
    model = si2D(frames)
    model.separate(frames)
    np.testing.assert_allclose(model.separr[0], 7, rtol=1e-6)
    np.testing.assert_allclose(model.separr[1], np.broadcast_to(2 * np.exp(-1j * theta), (128, 128)), atol=1e-6)
    np.testing.assert_allclose(model.separr[2], model.separr[1].conj())


def test_refinement_stops_when_search_does_not_converge():
    # A deliberately drifting search isolates the iteration-bound contract.
    class DriftingSearch:
        r_ang, r_sp = 0.08, 0.02

        def mapoverlap2(self, angle, spacing, **kwargs):
            return angle + 0.01, spacing, 1.0

    model = DriftingSearch()
    with pytest.raises(RuntimeError, match="did not converge"):
        runner.refine_parameters(model, 0.0, 0.7, index=0, max_iterations=2)
    assert (model.r_ang, model.r_sp) == (0.08, 0.02)


def test_enter_confirms_peak_and_closes_its_figure():
    selector = AngleSpacingCalculator(np.ones((128, 128)), dx=0.135)
    selector.get_ang_spacing_given_brightest_pixel((40, 64))
    number = selector.fig.number
    selector.on_key_press(SimpleNamespace(key="enter"))
    assert not plt.fignum_exists(number)


def test_refinement_and_phase_estimation_on_known_grating():
    rows = np.indices((128, 128))[0]
    frames = np.array([100 + 50 * np.cos(2 * np.pi * rows * 0.135 / 0.72 + 0.7 + phase)
                       for phase in [0, 2 * np.pi / 3, 4 * np.pi / 3]])
    model = si2D(frames)
    model.dx, model.na = 0.135, 0.8
    model.psf = model.getpsf()
    model.Ns = 2  # Exercise both search stages with a small grid.
    angle, spacing = runner.refine_parameters(model, 0.0, 0.72, index=0, max_iterations=3)
    assert angle == pytest.approx(0.0, abs=1e-8)
    assert spacing == pytest.approx(0.72)
    phase, _ = model.viewoverlap(angle, spacing, ind=0)
    assert phase == pytest.approx(-0.7, abs=1e-6)


def test_cli_fixed_parameters_writes_doubled_image(tmp_path):
    source = tmp_path / "input.tif"
    destination = tmp_path / "reconstructed"
    tifffile.imwrite(source, np.full((9, 128, 128), 100, dtype=np.uint16), photometric="minisblack")
    command = [sys.executable, "sim_recon_2d.py", str(source), "--output", str(destination),
               "--angles", "0", "2.0943951024", "4.1887902048", "--spacings", "0.7", "0.7", "0.7",
               "--phases", "0", "0", "0", "--no-refine", "--save-spectrum"]
    result = subprocess.run(command, cwd=Path(runner.__file__).parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    reconstructed = tifffile.imread(destination / "recon0.tif")
    assert reconstructed.shape == (256, 256)
    assert reconstructed.dtype == np.float32
    assert np.isfinite(reconstructed).all()
    assert (destination / "spectrum0.tif").is_file()
    assert (destination / "parameters.json").is_file()
    repeated = subprocess.run(command, cwd=Path(runner.__file__).parent, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "empty" in repeated.stderr.lower()
