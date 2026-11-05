"""Command-line entry point for three-angle, three-phase 2D SIM reconstruction."""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import tifffile

LOGGER = logging.getLogger(__name__)


def read_img(img_path):
    """Read TIFF or HDF5 root datasets named 0, 1, ..., in numeric order."""
    path = Path(img_path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        return tifffile.imread(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            keys = sorted((key for key in handle if key.isdecimal()), key=int)
            if not keys or [int(key) for key in keys] != list(range(len(keys))):
                raise ValueError("HDF5 needs consecutive root datasets named 0, 1, 2, ...")
            if any(not isinstance(handle[key], h5py.Dataset) for key in keys):
                raise ValueError("Numbered HDF5 entries must be datasets, not groups.")
            return np.stack([handle[key][...] for key in keys])
    raise ValueError(f"Unsupported input format: {path.suffix}; use TIFF or HDF5.")


def group_frames(images):
    """Return (planes, 9, height, width), with phase varying fastest."""
    images = np.asarray(images)
    if images.ndim not in (3, 4) or (images.ndim == 4 and images.shape[1] != 9):
        raise ValueError("Expected (9 * planes, H, W) or (planes, 9, H, W).")
    height, width = images.shape[-2:]
    if height != width or height < 2 or height % 2:
        raise ValueError("Images must be square with an even side length.")
    if images.size == 0 or images.size // (height * width) % 9:
        raise ValueError("Each plane must have exactly 9 frames (3 angles x 3 phases).")
    if not np.issubdtype(images.dtype, np.number) or np.iscomplexobj(images):
        raise ValueError("Input intensities must be real numeric values.")
    if not np.isfinite(images).all():
        raise ValueError("Input contains NaN or infinite intensities.")
    return images.reshape(-1, 9, height, width)


def refine_parameters(model, angle, spacing, index, max_iterations):
    """Repeat the original overlap grid search, then one finer search."""
    initial_angle_range, initial_spacing_range = model.r_ang, model.r_sp
    try:
        for _ in range(max_iterations):
            if spacing <= model.r_sp:
                raise ValueError("Spacing is too small for the overlap search range.")
            out = model.mapoverlap2(angle, spacing, ind=index, marr=True)
            if np.round(angle, 3) == np.round(out[0], 3) and np.round(spacing, 3) == np.round(out[1], 3):
                break
            angle, spacing = out[:2]
        else:
            raise RuntimeError(f"Angle {index + 1}: search did not converge in {max_iterations} iterations.")
        model.r_ang /= 4
        model.r_sp /= 4
        out = model.mapoverlap2(angle, spacing, ind=index, marr=True)
        return float(out[0]), float(out[1])
    finally:
        model.r_ang, model.r_sp = initial_angle_range, initial_spacing_range


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="TIFF stack or numbered HDF5 datasets")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"), help="New or empty output directory")
    parser.add_argument("--dx", type=float, default=0.135, help="Input pixel size in micrometers (default: 0.135)")
    parser.add_argument("--na", type=float, default=0.8, help="Numerical aperture (default: 0.8)")
    parser.add_argument("--wavelength", type=float, default=0.515, help="Emission wavelength in micrometers (default: 0.515)")
    parser.add_argument("--mu", type=float, default=0.2, help="Wiener parameter (default: 0.2)")
    parser.add_argument("--magnitudes", type=float, nargs=3, default=[1.8] * 3, help="Three sideband weights (default: 1.8 1.8 1.8)")
    parser.add_argument("--angles", type=float, nargs=3, help="Three initial directions in radians; otherwise select peaks interactively")
    parser.add_argument("--spacings", type=float, nargs=3, help="Three initial pattern periods in micrometers; required with --angles")
    parser.add_argument("--phases", type=float, nargs=3, help="Fixed phases in radians; requires --angles, --spacings and --no-refine")
    parser.add_argument("--no-refine", action="store_true", help="Use initial angles and spacings directly")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum coarse searches per direction (default: 20)")
    parser.add_argument("--max-planes", type=int, help="Process only the first N planes")
    parser.add_argument("--save-spectrum", action="store_true", help="Save the centered reconstructed Fourier magnitude")
    parser.add_argument("--show-plots", action="store_true", help="Show diagnostic plots after each plane")
    return parser


def reconstruct(args):
    # Delay GUI imports until after command-line parsing.
    import matplotlib
    if args.angles is not None and not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from get_angle_spacing import AngleSpacingCalculator
    from si_recon_2D_h import phaseest, si2D

    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("Output directory must be new or empty; choose a different --output path.")
    planes = group_frames(read_img(args.input))
    if args.max_planes is not None:
        planes = planes[:args.max_planes]
    initial_angles, initial_spacings = args.angles, args.spacings
    metadata = {
        "input_name": args.input.name,
        "dx_um": args.dx, "na": args.na, "wavelength_um": args.wavelength,
        "mu": args.mu, "magnitudes": args.magnitudes,
        "refine": not args.no_refine, "max_iterations": args.max_iterations,
        "output_pixel_size_um": args.dx / 2, "planes": [],
    }
    for index, plane in enumerate(planes):
        LOGGER.info("Reconstructing plane %d/%d", index + 1, len(planes))
        model = si2D(plane, show_plots=args.show_plots)
        model.dx, model.na, model.wavelength, model.mu = args.dx, args.na, args.wavelength, args.mu
        model.psf = model.getpsf()
        if initial_angles is None:
            initial_angles, initial_spacings = [], []
            for direction, spectrum in enumerate(phaseest(plane)):
                LOGGER.info("Select a noncentral frequency peak for direction %d, then press Enter.", direction + 1)
                selector = AngleSpacingCalculator(spectrum, dx=args.dx)
                selector.show()
                angle, spacing = selector.get_ang_spacing()
                initial_angles.append(angle)
                initial_spacings.append(spacing)
        angles, spacings = list(initial_angles), list(initial_spacings)
        phases = []
        for direction in range(3):
            if not args.no_refine:
                LOGGER.info("Refining direction %d/3", direction + 1)
                angles[direction], spacings[direction] = refine_parameters(
                    model, angles[direction], spacings[direction], direction, args.max_iterations)
            if args.phases is None:
                phase, _ = model.viewoverlap(angles[direction], spacings[direction], ind=direction)
                if not np.isfinite(phase):
                    raise ValueError(f"Plane {index}, direction {direction}: phase estimation failed.")
                phases.append(float(phase))
            else:
                phases.append(args.phases[direction])
        model.recon(angles, spacings, args.magnitudes, phases)
        reconstructed = np.abs(model.finalimage).astype(np.float32)
        if not np.isfinite(reconstructed).all():
            raise ValueError(f"Plane {index}: reconstruction contains nonfinite values.")
        args.output.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(args.output / f"recon{index}.tif", reconstructed, photometric="minisblack")
        if args.save_spectrum:
            spectrum = np.abs(np.fft.fftshift(model.S)).astype(np.float32)
            tifffile.imwrite(args.output / f"spectrum{index}.tif", spectrum, photometric="minisblack")
        metadata["planes"].append({"index": index, "angles_rad": angles, "spacings_um": spacings, "phases_rad": phases})
        (args.output / "parameters.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.show_plots:
            plt.show()
        plt.close("all")
    LOGGER.info("Saved %d reconstructed planes to %s", len(planes), args.output)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.angles is None) != (args.spacings is None):
        parser.error("--angles and --spacings must be supplied together.")
    if args.phases is not None and (args.angles is None or not args.no_refine):
        parser.error("--phases requires --angles, --spacings and --no-refine.")
    positives = [args.dx, args.na, args.wavelength, args.mu, *args.magnitudes, *(args.spacings or [])]
    if any(not np.isfinite(value) or value <= 0 for value in positives):
        parser.error("Pixel size, NA, wavelength, mu, magnitudes and spacings must be finite and positive.")
    if any(not np.isfinite(value) for value in [*(args.angles or []), *(args.phases or [])]):
        parser.error("Angles and phases must be finite.")
    if args.max_iterations < 1 or (args.max_planes is not None and args.max_planes < 1):
        parser.error("--max-iterations and --max-planes must be positive integers.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        reconstruct(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
