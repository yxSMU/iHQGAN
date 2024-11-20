import os
import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


import os
import csv
import glob
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim


def ssim(img1, img2):
    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = ((img1 - mu1) ** 2).mean()
    sigma2 = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    luminance = (2 * mu1 * mu2 + c1) / (mu1 ** 2 + mu2 ** 2 + c1)
    contrast = (2 * np.sqrt(sigma1) * np.sqrt(sigma2) +
                c2) / (sigma1 + sigma2 + c2)
    structure = (sigma12 + c2 / 2) / (np.sqrt(sigma1)
                                      * np.sqrt(sigma2) + c2 / 2)
    fsim_value = luminance * contrast * structure

    return fsim_value


def Metric_Calculate(gen_dir, real_dir):
    folder1_path = gen_dir
    folder2_path = real_dir
    num_images = 250

    image_files_folder1 = glob.glob(os.path.join(folder1_path, '*.png'))
    # image_files_folder1.sort(key=os.path.getmtime)
    # last_500_images_folder1 = image_files_folder1[-num_images:]

    image_files_folder2 = glob.glob(os.path.join(folder2_path, '*.png'))
    # image_files_folder2.sort(key=os.path.getmtime)
    # last_500_images_folder2 = image_files_folder2[-num_images:]

    fid_values = []
    psnr_value = []
    ssim_value = []

    for i in range(num_images):
        image_path_1 = image_files_folder1[i]
        image_path_2 = image_files_folder2[i]

        image1 = Image.open(image_path_1).convert("RGB")
        image2 = Image.open(image_path_2).convert("RGB")

        image1 = image1.resize((32, 32))
        image2 = image2.resize((32, 32))
        image1_gray = image1.convert("L")
        image2_gray = image2.convert("L")

        image1_array = np.array(image1_gray)
        image2_array = np.array(image2_gray)

        ms_ssim_value = ssim(image1_array, image2_array)
        ssim_value.append(ms_ssim_value)

        psnr = peak_signal_noise_ratio(image2_array, image1_array)
        psnr_value.append(psnr)

        image1_array = np.array(image1)
        image2_array = np.array(image2)

        image1_array = image1_array.flatten()
        image2_array = image2_array.flatten()

    average_ssim = np.mean(ssim_value)
    average_psnr_value = np.mean(psnr_value)

    formatted_ssim = f'{average_ssim:.2f}'
    formatted_psnr = f'{average_psnr_value:.2f}'
    return float(formatted_ssim), float(formatted_psnr)


Save_ssim = []
Save_psnr = []

for i in range(198, 5000, 200):

    A = f'F:\canny0critic_archi\\{i}\\result_B2A_Denoise'
    B = r'F:\ESWA\cyclegan\canny0\canny0198\A_real'
    

    Metric_Calculate(A, B)
    formatted_ssim, formatted_psnr = Metric_Calculate(A, B)
    Save_ssim.append(formatted_ssim)
    Save_psnr.append(formatted_psnr)


# ssim_values_rounded = np.round(Save_ssim, 2)
# np.save('ssim_canny0critic_1_10_150_B2A.npy', ssim_values_rounded)

print("SSIM:", Save_ssim)
print("PSNR:", Save_psnr)


max_ssim_value = np.max(Save_ssim)
max_ssim_index = np.argmax(Save_ssim)

max_psnr_value = np.max(Save_psnr)
max_psnr_index = np.argmax(Save_psnr)

print("SSIM_MAx:", max_ssim_value)
print("SSIM_MAxLocation:", max_ssim_index)

print("PSNR_MAx:", max_psnr_value)
print("PSNR_MAxLocation:", max_psnr_index)
