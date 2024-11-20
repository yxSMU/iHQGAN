import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
from UntilClassical.datasets_2_padding import ImageDataset
from torch.utils.data import Subset, DataLoader
import itertools
from torch.optim import Adam
from torchvision.utils import save_image
from UntilClassical.dataset import denorm
from UntilClassical.datasets_2_padding import ImageDataset
import os
import numpy as np

from models.CQCC_wganRes import ClassicalGAN1
from models.CQCC_wganRes2 import ClassicalGAN2
import random

def main():
   
    iter_num = 50
    out_dir = r'F:\cyclegan\one2one\dilate1\2dilate4_10C_16K22\A2B'
    out_dir_2 = r'F:\cyclegan\one2one\dilate1\2dilate4_10C_16K22\B2A'
    out_dir_3 = r'F:\cyclegan\one2one\dilate1\2dilate4_10C_16K22\A_real'
    out_dir_4 = r'F:\cyclegan\one2one\dilate1\2dilate4_10C_16K22\B_real'


    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_2, exist_ok=True)
    os.makedirs(out_dir_3, exist_ok=True)
    os.makedirs(out_dir_4, exist_ok=True)


    dataset_st = r"E:\Acdamic\Data\DataMake\MySelfData\Dilate4_shuffle"
    data = ImageDataset(dataset_st)
    dloader = DataLoader(data, batch_size=10,
                         shuffle=False, num_workers=1)  

    b1 = 0
    b2 = 0.999
    lr_D = 0.0002
    lr_G = 0.0002

    gan_1 = ClassicalGAN1()
    generator_1 = gan_1.Generator()
    critic_1 = gan_1.Discriminator()

    gan_2 = ClassicalGAN2()
    critic_2 = gan_2.Discriminator()

    total_params = sum(p.numel()
                       for p in generator_1.parameters() if p.requires_grad)
    print("Total number of parameters in the quantum generator: ", total_params)

    optimizer_G = torch.optim.Adam(itertools.chain(generator_1.parameters()),lr=lr_G,
                                   betas=(b1, b2))

    optimizer_D_1 = Adam(critic_1.parameters(), lr=lr_D, betas=(b1, b2))  
    optimizer_D_2 = Adam(critic_2.parameters(), lr=lr_D, betas=(b1, b2))  

    loss_criticA_history = []  
    loss_criticB_history = []  

    loss1_history = []  
    loss2_history = []  

    lossD1sum100 = 0
    lossD2sum100 = 0

    lossG1sum50 = 0
    lossG2sum50 = 0

    aver_loss_criticA_history = []
    aver_loss_criticB_history = []

    aver_loss1_history = []
    aver_loss2_history = []

    batches_done = 0
    criterion_cycle = torch.nn.L1Loss()
    MSE = nn.MSELoss()
    L1 = nn.L1Loss()

    batches_done = 0

    batch_indices_to_save = list(range(198, 5000, 200)) 
    batches_done = 0 

    for epoch in range(iter_num):
        for i, batch in enumerate(dloader):

            real_A = batch['A']
            real_B = batch['B']

            label = Variable(torch.ones(1))
            label2 = Variable(torch.ones(1))

            optimizer_D_1.zero_grad()
            fake_images_B = generator_1(real_A)
            real_validity_B = critic_1(real_B)
            real_loss = torch.mean((real_validity_B - label)**2)

            fake_validity_B = critic_1(fake_images_B)
            fake_loss = torch.mean(fake_validity_B**2)
            loss_critic_A = (real_loss + fake_loss)*0.5

            loss_criticA_history.append(loss_critic_A.item())
            lossD1sum100 = lossD1sum100 + loss_critic_A.item()

            loss_critic_A.backward()
            optimizer_D_1.step()

            # 2
            optimizer_D_2.zero_grad()
            fake_images_A = generator_1(real_B)
            real_validity_A = critic_2(real_A)
            real_loss1 = torch.mean((real_validity_A - label2)**2)

            fake_validity_A = critic_2(fake_images_A)
            fake_loss1 = torch.mean(fake_validity_A**2)
            loss_critic_B = (real_loss1 + fake_loss1)*0.5

            lossD2sum100 = lossD2sum100 + loss_critic_B.item()
            loss_critic_B.backward()

            optimizer_D_2.step()

            np.save(os.path.join(out_dir, 'loss_criticA_history.npy'),
                    loss_criticA_history)
            np.save(os.path.join(out_dir, 'loss_criticB_history.npy'),
                    loss_criticB_history)

            optimizer_G.zero_grad()
            pred_images_B = generator_1(real_A)
            fake_validity_B = critic_1(pred_images_B)
            g_loss_1 = torch.mean((fake_validity_B-label)**2)
            image_aba = generator_1(pred_images_B)
        
            loss_aba = L1(real_A, image_aba)*10

            gloss_a2b = g_loss_1 + loss_aba

            loss1_history.append(gloss_a2b.item())
            lossG1sum50 = lossG1sum50 + gloss_a2b.item()

            loss1_history.append(gloss_a2b.item())
            gloss_a2b.backward()
            optimizer_G.step()
            
            
            optimizer_G.zero_grad()
            pred_images_A = generator_1(real_B)
            fake_validity_A = critic_2(pred_images_A)
            g_loss_2 = torch.mean((fake_validity_A-label2)**2)

            image_bab = generator_1(pred_images_A)
            loss_bab = L1(real_B, image_bab)*10
            gloss_b2a = g_loss_2 + loss_bab

            loss2_history.append(gloss_b2a.item())
            lossG2sum50 = lossG2sum50 + gloss_b2a.item()

            gloss_b2a.backward()
            optimizer_G.step()

            save_image(denorm(pred_images_B), os.path.join(
                out_dir, '{}.png'.format(batches_done)), nrow=5)
            save_image(denorm(real_A), os.path.join(
                out_dir_3, '{}.png'.format(batches_done)), nrow=5)

            save_image(denorm(pred_images_A), os.path.join(
                out_dir_2, '{}.png'.format(batches_done)), nrow=5)
            save_image(denorm(real_B), os.path.join(
                out_dir_4, '{}.png'.format(batches_done)), nrow=5)

            print("saved images and state")
            np.save(os.path.join(out_dir, 'loss1_history.npy'), loss1_history)
            np.save(os.path.join(out_dir, 'loss2_history.npy'), loss2_history)

            print(
                f"[Epoch {epoch}/{iter_num}] [Batch {i}/{len(dloader)}] [D1 loss: {loss_critic_A.item()}] [D2 loss: {loss_critic_B.item()}] [G1 loss: {gloss_a2b.item()}][G2 loss: {gloss_b2a.item()}] ")

            batches_done += 1

            if batches_done in batch_indices_to_save:
                torch.save(generator_1.state_dict(), os.path.join(
                    out_dir, f'generator-batch{batches_done}.pt'))
               
        aver_loss_criticA = lossD1sum100 / 100
        aver_loss_criticB = lossD2sum100 / 100

        aver_loss1 = lossG1sum50 / 50
        aver_loss2 = lossG2sum50 / 50

        print(
            f"[Epoch {epoch}/{iter_num}][D1 loss: {aver_loss_criticA}] [D2 loss: {aver_loss_criticB}] [G1 loss: {aver_loss1}][G2 loss: {aver_loss2}]")

        aver_loss_criticA_history.append(aver_loss_criticA)
        aver_loss_criticB_history.append(aver_loss_criticB)

        aver_loss1_history.append(aver_loss1)
        aver_loss2_history.append(aver_loss2)

        np.save(os.path.join(out_dir, 'aver_loss_criticA_history.npy'),
                aver_loss_criticA_history)
        np.save(os.path.join(out_dir, 'aver_loss_criticB_history.npy'),
                aver_loss_criticB_history)

        np.save(os.path.join(out_dir, 'aver_loss1_history.npy'),
                aver_loss1_history)
        np.save(os.path.join(out_dir, 'aver_loss2_history.npy'),
                aver_loss2_history)

        lossD1sum100 = 0
        lossD2sum100 = 0

        lossG1sum50 = 0
        lossG2sum50 = 0

if __name__ == '__main__':
    main()
