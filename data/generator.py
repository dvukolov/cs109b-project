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
random_seed = 1314662
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
    # data_raw["psf_img"] = RawArray(ctypes.c_float, size * image_size ** 2)
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


def generate_sample(args):
    """Generate one valid sample and write it to the destination arrays.

    Args (packed as a tuple):
        i: index of the current sample
        sersic_index: an optional value of the Sersic Index (default: None, i.e. random)
        psf_re: an optional PSF (default: None, i.e. random)
        noise: an optional Gaussian noise level (default: None, i.e. random)
    """
    # Unpack the arguments
    i, sersic_index, psf_re, noise = args
    global counter

    # Loop until a sample satisfies all criteria
    while True:

        with counter.get_lock():
            # Increment the global iteration counter
            counter.value += 1
            # Initialize the random number generators
            random.seed(random_seed + counter.value)
            rng = galsim.BaseDeviate(random_seed + counter.value + 1)

        # SF moffat scale radius in arcsec: fixed vs random
        psf_re = random.uniform(0.5, 1) if psf_re is None else psf_re

        # Gaussian noise level: fixed vs random
        noise = random.randint(200, 400) if noise is None else noise

        # Sersic index: discrete vs continuous
        bulge_n = random.uniform(0.5, 6) if sersic_index is None else sersic_index

        # Sersic radius, unit arcsec
        bulge_re = random.uniform(0.1, 0.6)

        # q is ellipticity and beta is orientation.
        # You could directly predict q and beta but there would be a discontiniuty issue
        # for beta. A jump from 180 degree to 1 degree.
        # radial sampling for g1 and g2 -reduced shear -> ellipticiy and orientation
        A = random.uniform(0, 0.67)  # gal_q =b/a will ranges in (0.2,1) & A=1-q / 1+q
        gal_q = (1 - A) / (1 + A)
        gal_beta = random.uniform(0, 3.14)  # radians
        g_1 = A * np.cos(2 * gal_beta)
        g_2 = A * np.sin(2 * gal_beta)

        gal_flux = 1e5 * random.uniform(0.3, 4)

        gal = galsim.Sersic(bulge_n, half_light_radius=bulge_re)
        gal = gal.withFlux(gal_flux)
        gal = gal.shear(g1=g_1, g2=g_2)
        psf = galsim.Moffat(beta=psf_beta, flux=1.0, fwhm=psf_re)
        final = galsim.Convolve([psf, gal])
        image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
        final.drawImage(image=image)

        # signal to noise ratio
        snr = np.sqrt((image.array ** 2).sum()) / noise

        # After generating the data, preserve only that with SNR [10, 100]
        if 10 <= snr <= 100:
            break

    image_nonoise = copy.deepcopy(image.array)
    image.addNoise(galsim.PoissonNoise(rng, sky_level=0.0))
    # noise map for bkgr gaussian noise
    image.addNoise(galsim.GaussianNoise(rng, sigma=noise))

    # Optionally: generate a PSF image
    # psf_image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
    # psf.drawImage(image=psf_image)

    data["img"][i] = image.array  # final noised image
    data["img_nonoise"][i] = image_nonoise  # noiseless image
    # Optionally: save the PSF image
    # data['psf_img'][i] = psf_image.array
    data["gal_flux"][i] = gal_flux
    data["bulge_re"][i] = bulge_re
    data["bulge_n"][i] = bulge_n
    data["gal_q"][i] = gal_q
    data["gal_beta"][i] = gal_beta
    data["psf_r"][i] = psf_re
    data["snr"][i] = snr
    data["sigma"][i] = noise
    data["g_1"][i] = g_1
    data["g_2"][i] = g_2


def initialize_sersics(sersics, size):
    """Return either equally spaces discrete Sersic indices or None's

    Args:
        sersics: command line arguments for the number of discrete Sersic indices
        size: the number of samples to generate

    Returns:
        sersic_index: a vector of discrete Sersic indices or None's
    """
    # Define the Sersic indices
    if sersics is None:
        # Use random continuous Sersic indices
        sersic_index = [None] * size
    else:
        # Use a limited set of equally spaced Sersic indices
        assert size % sersics == 0
        sersic_index = np.linspace(0.5, 6, num=sersics)
        sersic_index = np.tile(sersic_index, size // sersics)
        np.random.seed(random_seed)
        np.random.shuffle(sersic_index)
    return sersic_index


def print_configuration(size, sersics, psf, noise, n_jobs):
    """Show the currently used configuration.
    """
    print("Generating galaxy images with the following parameters:")
    print("    Number of samples:", f"{size:,}")
    print(
        "    Sersic indices:",
        "random [0.5, 6.0]" if sersics is None else f"{sersics:,} discrete [0.5, 6.0]",
    )
    print("    PSF:", "random [0.5, 1.0]" if psf is None else psf)
    print("    Gaussian noise level:", "random [200, 400]" if noise is None else noise)
    print("    Signal-to-noise ratio: [10, 100]")
    print(f"    Starting random seed: {random_seed:,}")
    print("    Number of parallel jobs:", n_jobs)


@click.command()
@click.argument("filename")
@click.option("--size", default=10_000, help="Number of samples", show_default=True)
@click.option(
    "--sersics",
    default=None,
    type=click.IntRange(min=1, max=1000),
    help="Number of discrete Sersic indices between 1 and 1000 [default: random]",
)
@click.option(
    "--psf",
    default=None,
    type=click.FloatRange(min=0.5, max=1.0),
    help="The PSF radius between 0.5 and 1.0 [default: random]",
)
@click.option(
    "--noise",
    default=None,
    type=click.FloatRange(min=200, max=400),
    help="Gaussian noise level between 200 and 400 [default: random]",
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
def main(filename, size, sersics, psf, noise, seed, jobs):
    # Set the starting random seed
    global random_seed
    random_seed = seed

    # Allocate shared arrays
    allocate_memory(size)

    # Prepare the arguments
    i = np.arange(size)
    sersic_index = initialize_sersics(sersics, size)
    psf_list = [psf] * size
    noise_list = [noise] * size
    args = zip(i, sersic_index, psf_list, noise_list)

    # Unless specified, set the number of jobs to the number of CPU cores
    n_jobs = psutil.cpu_count(logical=False) if jobs is None else jobs
    # Show configuration
    print_configuration(size, sersics, psf, noise, n_jobs)

    # Generate the images
    with Pool(n_jobs) as pool:
        _ = list(tqdm(pool.imap(generate_sample, args), total=size, smoothing=0.01))

    print(f"Filtered out {counter.value - size:,} images based on extreme SNR")
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
        snr=data["snr"],
        sigma=data["sigma"],
    )
    print("Done")


if __name__ == "__main__":
    main()
