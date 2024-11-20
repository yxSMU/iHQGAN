import cv2
import os

input_folder = r'E:\ACMMM\YJC\YJC\TestResult\QRGAN_test\Noise\2\result_A_real'
output_folder =r'E:\ACMMM\YJC\YJC\TestResult\a\2'

if not os.path.exists(output_folder):
    os.makedirs(output_folder) 
kernel_size = (5,5)
sigma_x = 0

for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        image = cv2.imread(os.path.join(input_folder, filename))
        denoised_image = cv2.GaussianBlur(image, kernel_size, sigma_x)
        cv2.imwrite(os.path.join(output_folder, filename), denoised_image)
        print(1)