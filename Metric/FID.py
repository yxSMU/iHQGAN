import os
import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    epsilon = 1e-6
    sigma1 += epsilon * np.eye(sigma1.shape[0])
    sigma2 += epsilon * np.eye(sigma2.shape[0])

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def load_images_as_vectors(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize((28, 28))
                img_array = np.array(img).astype(np.float32)
                img_array = img_array / 255.0
                images.append(img_array)
            except Exception as e:
                print(f"cant handle {filename}: {e}")
    return np.array(images)



real_images_folder = f'F:\TestResult\Ablation\canny0critic_archi\\{{}}\\result_B2A_Denoise'
generated_images_folder = r'F:\canny0\canny0198\A_real'


fid_values = []
folder_indices = []


for i in range(198, 5000, 200):  
    real_folder_path = real_images_folder.format(i)
    generated_folder_path = generated_images_folder

    real_images = load_images_as_vectors(real_folder_path)
    generated_images = load_images_as_vectors(generated_folder_path)

    if real_images.size == 0 or generated_images.size == 0:
        print(f"cant find images in  {real_folder_path}")
        continue
   
    real_features = real_images.reshape(real_images.shape[0], -1)  
    generated_features = generated_images.reshape(
        generated_images.shape[0], -1)  
  
    fid_value = calculate_fid(real_features, generated_features)


    fid_values.append(fid_value)
    folder_indices.append(i)


fid_values_rounded = np.round(fid_values, 2)
np.save('fid_values_canny0critic_1_10_150_B2A.npy', fid_values_rounded)

print("FID array:")
for fid in fid_values:
    print(f"{fid:.2f}")
min_fid = np.min(fid_values)
min_fid_index = np.argmin(fid_values)
print(f"minFID: {min_fid:.2f}")
print(f"minFIDindex: {min_fid_index}")


plt.figure(figsize=(10, 5))
plt.plot(folder_indices, fid_values, marker='o', linestyle='-', color='b')
plt.title('FID')
plt.xlabel('Filename')
plt.ylabel('FID')
plt.xticks(folder_indices)  
plt.grid()
plt.show()
