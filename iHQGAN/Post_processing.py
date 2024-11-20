import cv2
import os

def denoise_image(image_path):
    img = cv2.imread(image_path)
    for i in range(8):
        row_pixels = img[i]
        black_pixels = row_pixels == 255
        img[i][black_pixels] = 0

    for i in range(26, 32):
        row_pixels = img[i]
        black_pixels = row_pixels == 255
        img[i][black_pixels] = 0

    return img

def process_images(base_dir):
    for folder in [str(i) for i in range(198, 4999, 200)]:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            for result_type in ['result_A2B', 'result_B2A']:
                result_path = os.path.join(folder_path, result_type)
                if os.path.exists(result_path):
                    denoised_folder = os.path.join(folder_path, f'{result_type}_Denoise')
                    os.makedirs(denoised_folder, exist_ok=True)

                    for img_idx in range(250):
                        img_path = os.path.join(result_path, str(img_idx) + '.png')
                        if os.path.exists(img_path):
                            denoised_img = denoise_image(img_path)
                            denoised_img_path = os.path.join(denoised_folder, str(img_idx) + '.png')
                            cv2.imwrite(denoised_img_path, denoised_img)

    print("处理完成。")

base_dir = r'F:\Coode\TestResult2\canny7critic_10_20_300'
process_images(base_dir)