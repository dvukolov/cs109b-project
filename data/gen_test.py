#!/usr/bin/env python

import copy
import ctypes
import random
from multiprocessing import Pool, RawArray, Value

import click
import galsim
import numpy as np
import psutil
from tqdm import tqdm


# Fixed parameters
image_size = 64  # n x n pixels
pixel_scale = 0.23  # arcsec / pixel
random_seed = 1314662 * 2
psf_beta = 2  # moffat parameter

# Dictionaries of arrays shared between processes
data_raw, data = {}, {}

# Set up a global counter
counter = Value("i", 0)


def allocate_memory(size):
    """Create global NumPy arrays which are shared between processes

    Args:
        size: the number of samples to generate
    """
    data_raw["img"] = RawArray(ctypes.c_float, size * image_size ** 2)
    data_raw["img_nonoise"] = RawArray(ctypes.c_float, size * image_size ** 2)
    # Optional: the PSF image
    data_raw["psf_img"] = RawArray(ctypes.c_float, 1 * image_size ** 2)
    data_raw["gal_flux"] = RawArray(ctypes.c_float, size)
    data_raw["bulge_n"] = RawArray(ctypes.c_float, size)
    data_raw["bulge_re"] = RawArray(ctypes.c_float, size)
    data_raw["gal_q"] = RawArray(ctypes.c_float, size)
    data_raw["gal_beta"] = RawArray(ctypes.c_float, size)
    data_raw["psf_r"] = RawArray(ctypes.c_float, size)
    data_raw["snr"] = RawArray(ctypes.c_float, size)
    data_raw["sigma"] = RawArray(ctypes.c_float, size)
    data_raw["g_1"] = RawArray(ctypes.c_float, size)
    data_raw["g_2"] = RawArray(ctypes.c_float, size)

    for name, raw_array in data_raw.items():
        data[name] = np.ctypeslib.as_array(raw_array)
        if name in ["img", "img_nonoise"]:
            data[name] = data[name].reshape(size, image_size, image_size)
        elif name in ["psf_img"]:
            data[name] = data[name].reshape(1, image_size, image_size)


def generate_sample(args):
    """Generate one valid sample and write it to the destination arrays.
    """
    i, flux, bulge_n, bulge_re, g1, g2, psf_re, noise = args

    with counter.get_lock():
        # Increment the global iteration counter
        counter.value += 1
        # Initialize the random number generators
        random.seed(random_seed + counter.value)
        rng = galsim.BaseDeviate(random_seed + counter.value + 1)

    gal = galsim.Sersic(bulge_n, half_light_radius=bulge_re)
    gal = gal.withFlux(flux)
    gal = gal.shear(g1=g1, g2=g2)
    psf = galsim.Moffat(beta=psf_beta, flux=1.0, fwhm=psf_re)
    final = galsim.Convolve([psf, gal])
    image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
    final.drawImage(image=image)

    # signal to noise ratio
    snr = np.sqrt((image.array ** 2).sum()) / noise

    image_nonoise = copy.deepcopy(image.array)
    image.addNoise(galsim.PoissonNoise(rng, sky_level=0.0))
    # noise map for bkgr gaussian noise
    image.addNoise(galsim.GaussianNoise(rng, sigma=noise))

    # Optionally: generate a PSF image
    if i == 0:
        psf_image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
        psf.drawImage(image=psf_image)
        data["psf_img"][i] = psf_image.array

    data["img"][i] = image.array  # final noised image
    data["img_nonoise"][i] = image_nonoise  # noiseless image
    data["gal_flux"][i] = flux
    data["bulge_re"][i] = bulge_re
    data["bulge_n"][i] = bulge_n
    data["psf_r"][i] = psf_re
    data["snr"][i] = snr
    data["sigma"][i] = noise
    data["g_1"][i] = g1
    data["g_2"][i] = g2


def print_configuration(
    size, flux, sersic_index, sersic_radius, g1, g2, psf, noise, n_jobs
):
    """Show the currently used configuration.
    """
    print("Generating galaxy images with the following parameters:")
    print("    Number of samples:", f"{size:,}")
    print(f"    Flux: {flux:,}")
    print("    Sersic index:", sersic_index)
    print("    Sersic index:", sersic_radius)
    print("    g1:", g1)
    print("    g2:", g2)
    print("    PSF:", psf)
    print("    Gaussian noise level:", noise)
    print(f"    Starting random seed: {random_seed:,}")
    print("    Number of parallel jobs:", n_jobs)


@click.command()
@click.argument("filename")
@click.option("--size", default=10_000, help="Number of samples", show_default=True)
@click.option(
    "--flux",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.3, max=4.0),
    help="Flux in 10^5 between 0.3 and 4.0",
)
@click.option(
    "--sersic-index",
    default=3.0,
    show_default=True,
    type=click.FloatRange(min=0.5, max=6.0),
    help="Sersic index between 0.5 and 6.0",
)
@click.option(
    "--sersic-radius",
    default=0.3,
    show_default=True,
    type=click.FloatRange(min=0.1, max=0.6),
    help="Sersic radius in arcsec between 0.1 and 0.6",
)
@click.option(
    "--g1",
    default=-0.069,
    show_default=True,
    type=click.FloatRange(min=-0.67, max=0.67),
    help="g1 between -0.67 and 0.67",
)
@click.option(
    "--g2",
    default=0.15,
    show_default=True,
    type=click.FloatRange(min=-0.67, max=0.67),
    help="g2 between -0.67 and 0.67",
)
@click.option(
    "--psf",
    default=0.5,
    show_default=True,
    type=click.FloatRange(min=0.5, max=1.0),
    help="The PSF radius between 0.5 and 1.0",
)
@click.option(
    "--noise",
    required=True,
    type=click.FloatRange(min=200, max=400),
    help="Gaussian noise level between 200 and 400",
)
@click.option(
    "--seed",
    default=random_seed,
    show_default=True,
    type=click.IntRange(min=0),
    help="Starting value of the random seed",
)
@click.option(
    "--jobs",
    default=None,
    type=click.IntRange(min=1),
    help="Number of parallel processes to run [default: number of CPU cores]",
)
def main(
    filename, size, flux, sersic_index, sersic_radius, g1, g2, psf, noise, seed, jobs,
):
    # Set the starting random seed
    global random_seed
    random_seed = seed

    # Allocate shared arrays
    allocate_memory(size)

    # Prepare the arguments
    flux *= 10 ** 5
    args = [
        (i, flux, sersic_index, sersic_radius, g1, g2, psf, noise) for i in range(size)
    ]

    # Unless specified, set the number of jobs to the number of CPU cores
    n_jobs = psutil.cpu_count(logical=False) if jobs is None else jobs
    # Show configuration
    print_configuration(
        size, flux, sersic_index, sersic_radius, g1, g2, psf, noise, n_jobs
    )

    # Generate the images
    with Pool(n_jobs) as pool:
        _ = list(tqdm(pool.imap(generate_sample, args), total=size, smoothing=0.01))

    avg_snr = data["snr"].mean()
    std_snr = data["snr"].std()
    print(
        f"Average signal-to-noise ratio: {avg_snr:.2f}; standard deviation: {std_snr:.6f}"
    )

    print(f"Filtered out {counter.value - size:,} images based on extreme noise levels")
    print(f"Saving the data to {filename!r}")

    # Concatenate the labels: "Flux", "Sersic Index", "Sersic Radius", "g1", "g2"
    label = np.stack(
        (data["gal_flux"], data["bulge_n"], data["bulge_re"], data["g_1"], data["g_2"]),
        axis=1,
    )

    # Save the data
    np.savez(
        filename,
        img=data["img"],
        img_nonoise=data["img_nonoise"],
        label=label,
        psf_r=data["psf_r"],
        psf_img=data["psf_img"],
        snr=data["snr"],
        sigma=data["sigma"],
    )
    print("Done")


if __name__ == "__main__":
    main()
