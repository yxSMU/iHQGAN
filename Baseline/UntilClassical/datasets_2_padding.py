import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        # self.transform = transforms.Compose(transforms_)
        self.transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.unaligned = unaligned

        # self.files_A = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        # self.files_B = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_A = sorted(glob.glob(os.path.join(root, 'B', '*.png')))
        self.files_B = sorted(glob.glob(os.path.join(root, 'A', '*.png')))

    def add_padding(self, image):
        width, height = image.size
     
        new_width = width + 4
        new_height = height + 4
        new_image = Image.new('L', (new_width, new_height), color=0)
        new_image.paste(image, (2, 2))

        return new_image


    def __getitem__(self, index):
        item_A_add=self.add_padding(Image.open(self.files_A[index]))
        item_A = self.transform(item_A_add)

        item_B_add = self.add_padding(Image.open(self.files_B[index]))
        item_B = self.transform(item_B_add)

        return {'B': item_A, 'A': item_B}


    def __len__(self):
        return min(len(self.files_A), len(self.files_B))
