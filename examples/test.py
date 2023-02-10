#%%
import scipy as sp
from scipy import ndimage
import numpy as np
import skimage as skimg
import matplotlib.pyplot as plt
import os


os.chdir(os.path.dirname(__file__))

image=skimg.io.imread("../datasets/brain_tumor_dataset/no/N1.JPG", as_gray=True)
print(image.shape)

fig = plt.figure(figsize=(10, 10), dpi=80)
fig.suptitle("Denoising", fontsize=15)

plt.subplot(2,2,1)
plt.title('Original image')
plt.imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

size = 3
unifilt_img = ndimage.uniform_filter(image, size=size)
plt.subplot(2,2,2)
plt.title(f'Uniform filter, size={size}')
plt.imshow(unifilt_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

sigma = 1
gaussian_img = ndimage.gaussian_filter(image, sigma=sigma)
plt.subplot(2,2,3)
plt.title(f'Gaussian filter, sigma={sigma}')
plt.imshow(gaussian_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

size_median = 3
median_img = ndimage.median_filter(image, size=3)
plt.subplot(2,2,4)
plt.title(f'Median filter, size={size_median}')
plt.imshow(median_img,cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

sobel_img = ndimage.sobel(image, axis=1)
plt.figure()
plt.title('Sobel filter')
plt.imshow(median_img,cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

# %%
