# Perlin Noise Generator
# http://en.wikipedia.org/wiki/Perlin_noise
# http://en.wikipedia.org/wiki/Bilinear_interpolation
# FB36 - 20130222
# Code adapted from:
# https://code.activestate.com/recipes/578470-perlin-noise-generator/
from os.path import exists
from os import mkdir
import random
import numpy as np
import cv2

# Can load an image and apply the noise over it
# image = cv2.imread('IMAGE_NAME.png')
imgx, imgy, imgz = (256, 256, 3)

np.random.seed(1)

save_dir = 'textures'
if not exists(save_dir):
    mkdir(save_dir)

num_images = 200
for num in range(num_images):
    octaves = int(np.log2(max(imgx, imgy)))
    persistence = np.random.normal(0.9, 0.2)
    imgAr = np.zeros((imgx, imgy))
    totAmp = 0.0

    for k in range(octaves):
        freq = 2 ** k
        amp = persistence ** k
        totAmp += amp
        # create an image from n by m grid of random numbers (w/ amplitude)
        # using Bilinear Interpolation
        n = freq + 1; m = freq + 1 # grid size
        ar = [[random.random() * amp for i in range(n)] for j in range(m)]
        nx = imgx / (n - 1.0); ny = imgy / (m - 1.0)
        for ky in range(imgy):
            for kx in range(imgx):
                i = int(kx / nx); j = int(ky / ny)
                dx0 = kx - i * nx; dx1 = nx - dx0
                dy0 = ky - j * ny; dy1 = ny - dy0
                z = ar[j][i] * dx1 * dy1
                z += ar[j][i + 1] * dx0 * dy1
                z += ar[j + 1][i] * dx1 * dy0
                z += ar[j + 1][i + 1] * dx0 * dy0
                z /= nx * ny
                imgAr[ky][kx] += z # add image layers together

    noise = np.zeros((imgx, imgy, imgz))
    for channel in range(imgz):
        # image[:, :, channel] = np.multiply(image[:, :, channel], imgAr)
        noise[:, :, channel] = imgAr * np.random.normal(0.5, 0.45)

    # imax, imin = image.max(), image.min()
    # imin = 0.000000001 if imin == 0 else imin
    # image = (image - imin)/(imax - imin) * 255

    nmax, nmin = noise.max(), noise.min()
    nmin = 1e-7 if nmin == 0 else nmin
    noise = (noise - nmin)/(nmax - nmin) * 255


    # cv2.imwrite(f"{save_dir}/{num}_Perlin_Image.png", image)
    cv2.imwrite(f"{save_dir}/{num}_Perlin_Noise.png", noise)