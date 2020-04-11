import copy
import galsim
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Fixed parameters
image_size = 64  # n x n pixels
pixel_scale = 0.23  # arcsec / pixel
random_seed = 1314662
rng = galsim.BaseDeviate(random_seed + 1)
psf_beta = 2  # moffat parameter

# User-configurable parameters
params = {
    "PSF moffat scale radius (in arcsec)": dict(
        min=0.5, max=1.0, default=0.75, step=0.05, name="psf_re"
    ),
    "Sersic radius (in arcsec)": dict(
        min=0.1, max=0.6, default=0.35, step=0.05, name="bulge_re"
    ),
    "Sersic index": dict(min=0.5, max=6.0, default=3.25, step=0.25, name="bulge_n"),
    "Elipticity": dict(min=0.2, max=1.0, default=0.6, step=0.1, name="gal_q"),
    "Orientation (in radians)": dict(
        min=0.0, max=3.14, default=3.14 / 2, step=0.1, name="gal_beta"
    ),
    "Noise level": dict(min=200, max=400, default=300, step=10, name="noise"),
    "Flux": dict(
        min=0.3 * 1e5,
        max=4.0 * 1e5,
        default=2.15 * 1e5,
        step=0.1 * 1e5,
        name="gal_flux",
    ),
}


def generate_image(psf_re, bulge_re, bulge_n, gal_q, gal_beta, noise, gal_flux):
    A = (1 - gal_q) / (gal_q + 1)
    g_1 = A * np.cos(2 * gal_beta)
    g_2 = A * np.sin(2 * gal_beta)

    gal = galsim.Sersic(bulge_n, half_light_radius=bulge_re)
    gal = gal.withFlux(gal_flux)
    gal = gal.shear(g1=g_1, g2=g_2)
    psf = galsim.Moffat(beta=psf_beta, flux=1.0, fwhm=psf_re)
    final = galsim.Convolve([psf, gal])
    image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
    final.drawImage(image=image)
    image_nonoise = copy.deepcopy(image.array)

    # signal to noise ratio, after generating data, choose data with snr [10,100]
    snr = np.sqrt((image.array ** 2).sum()) / noise

    image.addNoise(galsim.PoissonNoise(sky_level=0.0))
    noisemap = np.random.normal(0, noise, 64 * 64)  # noise map for bkgr gaussian noise
    noisemap = noisemap.reshape((64, 64))
    img_fv = image.array + noisemap
    # you can also use add noise to add gaussian noise by using
    # 'image.addNoise(galsim.GaussianNoise(sigma=noise)) '

    final_2 = psf
    image_2 = galsim.ImageF(image_size, image_size, scale=pixel_scale)
    final_2.drawImage(image=image_2)

    outcome = {
        "noised": img_fv,
        "psf": image_2.array,
        "noiseless": image_nonoise,
        "snr": snr,
        "g1": g_1,
        "g2": g_2,
    }
    return outcome


def draw_images(outcome):
    images = outcome["noised"], outcome["noiseless"], outcome["psf"]
    titles = ["Noise Image", "Noiseless Image", "PSF"]

    fig, axes = plt.subplots(1, 3, figsize=(8, 4), constrained_layout=True)
    for image, title, ax in zip(images, titles, axes):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title)

    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def main():
    st.title("EDA of Simulated Galaxies")
    st.sidebar.title("Parameters")

    settings = {}
    for description, param in params.items():
        settings[param["name"]] = st.sidebar.slider(
            description, param["min"], param["max"], param["default"], param["step"]
        )

    outcome = generate_image(**settings)

    images = draw_images(outcome)
    st.image(images)
    st.write(
        f"Signal-to-Noise Ratio: {outcome['snr']:.2f}; g1: {outcome['g1']:.4f}, g2: {outcome['g2']:.4f}"
    )
    st.write(settings)


if __name__ == "__main__":
    main()
