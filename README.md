# PALS_SIM_recon

Python code for **2D structured illumination microscopy (SIM) reconstruction**, using three illumination directions and three phase steps per direction. Each group of nine raw frames is reconstructed independently onto a grid with twice the input width and height.

This is the reconstruction code accompanying an **Optica** research article.

This repository contains reconstruction code, documentation and synthetic tests. Experimental images, calibration files and reconstruction results are not included.

## Installation

Tested with Python 3.12. Create an environment, then install the pinned runtime dependencies:

```bash
git clone https://github.com/Paratra/PALS_SIM_recon.git
cd PALS_SIM_recon
python -m venv .venv
```

Activate it on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or on Linux/macOS with `source .venv/bin/activate`, then run:

```bash
python -m pip install -r requirements.txt
python sim_recon_2d.py --help
```

Interactive peak selection requires a Matplotlib GUI backend, such as TkAgg or QtAgg, and a display. With a minimal Python installation, install a Tk or Qt backend separately, or supply angles and spacings for headless execution.

## Input layout

- TIFF: `(9 * number_of_planes, height, width)` or `(number_of_planes, 9, height, width)`.
- HDF5: consecutive root datasets named `0`, `1`, `2`, ...; numeric order is used. Each dataset is one 2D frame or one `(9, height, width)` plane. Nonnumeric metadata entries are ignored.
- Images must be square, have an even side length, and contain finite real intensities.
- Within each plane, **phase varies fastest**:

```text
direction 0: phase 0, phase 1, phase 2
direction 1: phase 0, phase 1, phase 2
direction 2: phase 0, phase 1, phase 2
```

The separation matrix assumes equally spaced phase steps of `0`, `2π/3`, and `4π/3`. Multichannel data must be separated and reordered before use. A stack of optical sections is processed as independent 2D reconstructions; this is not a 3D SIM reconstruction model.

## Interactive reconstruction

```bash
python sim_recon_2d.py path/to/raw_stack.tif --output results/run_01
```

For each of the three directions in the first plane:

1. Drag a rectangle around a sideband peak in the Fourier image, away from the center.
2. Press **Enter** to confirm the brightest pixel in that rectangle and close the window.
3. Repeat for the next direction.

The initial angles and spacings are reused for subsequent planes. Each plane then receives its own overlap-based refinement and phase estimate. Use `--max-planes 1` for an initial trial.

## Reconstruction with supplied parameters

Providing both `--angles` and `--spacings` skips interactive selection. Values below are illustrative; substitute parameters calibrated for your acquisition:

```bash
python sim_recon_2d.py path/to/raw_stack.tif --output results/run_02 --angles 1.57 -0.46 0.48 --spacings 0.73 0.71 0.70
```

By default, supplied angles and spacings are starting guesses for refinement. Add `--no-refine` to retain them and estimate only the phases. To use fully fixed parameters, also provide all three phases:

```bash
python sim_recon_2d.py path/to/raw_stack.tif --output results/run_03 --angles 1.57 -0.46 0.48 --spacings 0.73 0.71 0.70 --phases 0.78 -2.24 -3.12 --no-refine
```

Fixed phases require `--no-refine`, since changing the pattern geometry can invalidate them. A single supplied parameter set applies to every processed plane. Without `--show-plots`, runs with supplied angles use a noninteractive backend.

## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--dx` | `0.135` | Input pixel size in µm |
| `--na` | `0.8` | Detection numerical aperture |
| `--wavelength` | `0.515` | Emission wavelength in µm |
| `--mu` | `0.2` | Wiener parameter; denominator includes `mu²` |
| `--magnitudes` | `1.8 1.8 1.8` | Sideband weights, one per direction |
| `--angles` | Interactive | Three directions in radians |
| `--spacings` | Interactive | Three pattern periods in µm |
| `--phases` | Estimated | Three phases in radians; fixed geometry only |
| `--max-iterations` | `20` | Maximum coarse refinement iterations per direction |
| `--max-planes` | All | Process the first N planes |
| `--save-spectrum` | Off | Save the reconstructed Fourier magnitude |
| `--show-plots` | Off | Show diagnostic figures after each plane |

Angles retain the original image-coordinate convention: top = `0`, left = `π/2`, bottom = `π`, right = `-π/2`, modulo `2π`. The defaults come from the original experiment scripts and should be set for the imaging system in use. The sideband weights remain fixed by default, as in the original runner; they are not automatically replaced by the overlap magnitude.

## Outputs

The output directory must be new or empty to prevent accidental overwrites.

- `recon0.tif`, `recon1.tif`, ...: absolute reconstructed intensity, `float32`, shape `(2H, 2W)`.
- `spectrum0.tif`, ...: optional `abs(fftshift(S))`, where `S` is the reconstructed Fourier image. These files are **not an optical transfer function**, despite the `otf` filenames used by the old scripts.
- `parameters.json`: optical settings, sideband weights and per-plane angles, spacings and phases. The output sampling interval is `dx / 2` and is recorded here; TIFF pixel calibration metadata is not written.

The full input stack is loaded into memory. Refinement uses repeated Fourier transforms and may take time on larger images. If refinement fails, the command reports the failure; outputs from already completed planes remain available.

## Code organization

| File | Purpose |
| --- | --- |
| `sim_recon_2d.py` | CLI, image loading, validation, parameter estimation and output |
| `si_recon_2D_h.py` | Phase separation, Fourier interpolation/shifting, overlap search and Wiener reconstruction |
| `get_angle_spacing.py` | Interactive peak selection and angle/spacing calculation |
| `psfsim_y_p36.py` | Original pupil, PSF and OTF model |
| `Utility_p36.py` | Circular pupils, radial arrays and wavefront helpers |
| `zernike_p36.py` | Original Zernike polynomial utilities |
| `tests/test_reconstruction.py` | Synthetic tests generated at runtime |

The original module names are retained for familiarity. Dataset-specific scripts, backups, hardcoded workstation paths, debug breakpoints and abandoned commented implementations were excluded or removed.

### Numerical compatibility

The original reconstruction expressions, Fourier normalization, search grid and PSF model are retained. CLI defaults match the original main script. Direct `si2D` construction retains its legacy defaults (`dx=0.089`, `na=1.20`, `mu=0.01`); after setting optical attributes, assign `model.psf = model.getpsf()` when using the low-level API.

One legacy PSF detail is deliberately preserved: `psf.setParams(nx=...)` does not update `ny`, while pupil-frequency sampling uses `ny`, initially `128`. Consequently, changing image size does not automatically rescale this pupil model as one might expect. This cleanup does not recalibrate the optical model; check its sampling before adapting it to a different acquisition.

The CLI adds bounded refinement, input checks, optional diagnostic plots and working peak confirmation. Numerical compatibility checks against the original code are distinct from validation of physical image quality.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Tests generate small synthetic arrays and temporary TIFF/HDF5 files; no experimental data is required. `.gitignore` excludes image files, numerical data, output directories and Python caches.

## Attribution

The core reconstruction code identifies **Peter Kner, University of Georgia (2019)** as its author. The PSF and Zernike utilities also originate from Peter Kner's code; the original Zernike copyright notice is retained. This repository organizes the existing project and its later experiment runner into a reusable command-line workflow.

## License

Released under the **[BSD 3-Clause License](LICENSE)**. It permits use, modification and redistribution, including commercial use, while requiring preservation of the copyright notice and license terms. The names of the copyright holders and contributors may not be used to endorse derived products without prior written permission. See `LICENSE` for the complete terms and warranty disclaimer.

## Citation

If you use this code in published research, please cite the associated Optica article and reference this repository, including the commit used for your results. Article citation details will be added when available.

Software citation metadata is provided in [CITATION.cff](CITATION.cff), which GitHub displays through its **Cite this repository** feature. The citation request is an academic courtesy, not an additional restriction on the BSD 3-Clause License.
