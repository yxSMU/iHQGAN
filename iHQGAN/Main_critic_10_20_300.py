import os
import argparse
import math
import numpy as np
import torch
import itertools
from torch.utils.data import Subset, DataLoader
from PIL import Image
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.utils import save_image
from utils.dataset import denorm
from utils.datasets_2_padding import ImageDataset
from utils.wgan import compute_gradient_penalty
from models.quantum_generator import PQWGAN_CC
from models.reverse_quantum_generator import PQWGAN_CC3
from models.acnn import ClassicalGAN1
from models.acnn2 import ClassicalGAN2
from tqdm import trange
import time
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
# import cv2
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm

def fsim(img1, img2):
    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = ((img1 - mu1) ** 2).mean()
    sigma2 = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    luminance = (2 * mu1 * mu2 + c1) / (mu1 ** 2 + mu2 ** 2 + c1)
    contrast = (2 * torch.sqrt(sigma1) * torch.sqrt(sigma2) + c2) / (sigma1 + sigma2 + c2)
    structure = (sigma12 + c2 / 2) / (torch.sqrt(sigma1) * torch.sqrt(sigma2) + c2 / 2)

    fsim_value = luminance * contrast * structure

    return fsim_value

def train(dataset_str, patches, layers, n_data_qubits, batch_size, checkpoint,
          patch_shape):
    device = torch.device('cpu')
    n_epochs = 50
    image_size = 32
    image_size_2 = 32
    channels = 1

    ancillas = 0
    if n_data_qubits:
        qubits = n_data_qubits + ancillas
    else:
        qubits = math.ceil(math.log(image_size ** 2 // patches, 2)) + ancillas

    # lr_D = 0.1
    # lr_G = 0.1
    lr_D = 0.0002
    lr_G = 0.01
    lr_G_classical = 0.0002

    b1 = 0
    b2 = 0.9
    lambda_gp = 10
    n_critic = 2

    out_dir = "TraninResult\canny0critic_10_20_300\\result_A2B"
    out_dir_2 = "TraninResult\canny0critic_10_20_300\\result_B2A"
    out_dir_3 = "TraninResult\canny0critic_10_20_300\\result_A_real"
    out_dir_4 = "TraninResult\canny0critic_10_20_300\\result_B_real"
    out_dir_5 = "TraninResult\canny0critic_10_20_300\\a2a"
    out_dir_6 = "TraninResult\canny0critic_10_20_300\\b2b"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_2, exist_ok=True)
    os.makedirs(out_dir_3, exist_ok=True)
    os.makedirs(out_dir_4, exist_ok=True)
    os.makedirs(out_dir_5, exist_ok=True)
    os.makedirs(out_dir_6, exist_ok=True)

    gan_1 = PQWGAN_CC(image_size=image_size, image_size_2=image_size_2, channels=1, n_generators=patches,
                      n_qubits=qubits,
                      n_ancillas=ancillas, n_layers=layers, patch_shape=patch_shape)

    gan_2 = PQWGAN_CC3(image_size=image_size, image_size_2=image_size_2, channels=1, n_generators=patches,
                       n_qubits=qubits,
                       n_ancillas=ancillas, n_layers=layers, patch_shape=patch_shape)

    critic = gan_1.critic  
    generator = gan_1.generator

    critic_2 = gan_2.critic  
    generator_2 = gan_2.generator


    gan_3 = ClassicalGAN1()
    generator_3 = gan_3.Generator()

    gan_4 = ClassicalGAN2()
    generator_4 = gan_4.Generator()

    total_params = sum(p.numel() for p in gan_1.generator.params if p.requires_grad)
    print("Total number of parameters in the quantum generator: ", total_params)

    generator_2.load_state_dict(generator.state_dict())

    data = ImageDataset(dataset_str)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)

    # optimizer_G = Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    # optimizer_G_2 = Adam(generator_2.parameters(), lr=lr_G, betas=(b1, b2))

    optimizer_G = Adam([
        {'params': generator.parameters(), 'lr': lr_G},
        {'params': generator_3.parameters(), 'lr': lr_G_classical}
    ], betas=(b1, b2))

    optimizer_G_2 = Adam([
        {'params': generator_2.parameters(), 'lr': lr_G},
        {'params': generator_4.parameters(), 'lr': lr_G_classical}
    ], betas=(b1, b2))


    optimizer_D_1 = Adam(critic.parameters(), lr=lr_D, betas=(b1, b2)) 
    optimizer_D_2 = Adam(critic_2.parameters(), lr=lr_D, betas=(b1, b2))  


    loss_criticA_history = [] 
    loss_criticB_history = [] 

    loss1_history = []  
    loss2_history = [] 

    lossD1sum100 = 0
    lossD2sum100 = 0

    lossG1sum50 = 0
    lossG2sum50 = 0

    lossW1sum50 = 0
    lossW2sum50 = 0

    aver_loss_criticA_history = []
    aver_loss_criticB_history = []

    aver_loss1_history = []
    aver_loss2_history = []

    batches_done = 0

    # if checkpoint != 0:
    #     critic.load_state_dict(torch.load(out_dir + f"/critic-{checkpoint}.pt"))
    #     generator.load_state_dict(torch.load(out_dir + f"/generator-{checkpoint}.pt"))

    #     critic_2.load_state_dict(torch.load(out_dir + f"/critic_2-{checkpoint}.pt"))
    #     generator_2.load_state_dict(torch.load(out_dir + f"/generator_2-{checkpoint}.pt"))

    #     wasserstein_distance_history = list(np.load(out_dir + "/wasserstein_distance.npy"))
    #     wasserstein_distance_history2 = list(np.load(out_dir + "/wasserstein_distance2.npy"))
    #     batches_done = checkpoint


    # start_epoch=15
    # start_batch=92
    for epoch in trange(n_epochs):
        for i, batch in tqdm(enumerate(dataloader)):
            # if epoch<start_epoch or (epoch==start_epoch and i<start_batch):
            #     continue
            real_A = batch['A']
            # Set model inputzzX
            real_B = batch['B']
            optimizer_D_1.zero_grad()
         
            fake_images_B = generator(real_A)
      
            real_validity_B = critic(real_B)
            fake_validity_B = critic(fake_images_B)
       
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_B, fake_images_B, device)
            # Adversarial loss
            loss_critic_A = -torch.mean(real_validity_B) + torch.mean(fake_validity_B) + lambda_gp * gradient_penalty
            loss_criticA_history.append(loss_critic_A.item())

            lossD1sum100 = lossD1sum100 + loss_critic_A.item()
            # Adversarial loss
            loss_critic_A.backward()
            optimizer_D_1.step()
       
            optimizer_D_2.zero_grad()
       
            fake_images_A = generator_2(real_B)
       
            real_validity_A = critic_2(real_A)
            fake_validity_A = critic_2(fake_images_A)

      
            # Gradient penalty
            gradient_penalty_2 = compute_gradient_penalty(critic_2, real_A, fake_images_A, device)
            # Adversarial loss
            loss_critic_B = -torch.mean(real_validity_A) + torch.mean(fake_validity_A) + lambda_gp * gradient_penalty_2
            loss_criticB_history.append(loss_critic_B.item())

            lossD2sum100 = lossD2sum100 + loss_critic_B.item()

            loss_critic_B.backward()
            optimizer_D_2.step()

            np.save(os.path.join(out_dir, 'loss_criticA_history.npy'), loss_criticA_history)
            np.save(os.path.join(out_dir, 'loss_criticB_history.npy'), loss_criticB_history)

            if i % n_critic == 0:
      
                optimizer_G.zero_grad()
                pred_images_B = generator(real_A)
                fake_validity_B = critic(pred_images_B)
              
                g_loss_1 = -torch.mean(fake_validity_B) * 10
              
                image_aba = generator_3(pred_images_B)
                loss_aba = F.l1_loss(real_A, image_aba) * 20

            
                fsim_value_1 = fsim(denorm(real_A), denorm(image_aba)) * 300
               
                gloss_a2b = g_loss_1 + loss_aba + fsim_value_1

                loss1_history.append(gloss_a2b.item())
                lossG1sum50 = lossG1sum50 + gloss_a2b.item()
               
                gloss_a2b.backward()
                optimizer_G.step()

                # print(loss_aba)
                # print(g_loss_1)

                generator_2.load_state_dict(generator.state_dict())

            
                save_image(denorm(pred_images_B), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
                save_image(denorm(real_A), os.path.join(out_dir_3, '{}.png'.format(batches_done)), nrow=5)
                save_image(denorm(image_aba), os.path.join(out_dir_5, '{}.png'.format(batches_done)), nrow=5)
                torch.save(critic.state_dict(), os.path.join(out_dir, 'critic-{}.pt'.format(batches_done)))
                torch.save(generator.state_dict(), os.path.join(out_dir, 'generator-{}.pt'.format(batches_done)))
                torch.save(generator_3.state_dict(), os.path.join(out_dir_5, 'generator_3-{}.pt'.format(batches_done)))
                print("saved images and state")
               
                optimizer_G_2.zero_grad()
                pred_images_A = generator_2(real_B)
                fake_validity_A = critic_2(pred_images_A)
             
                g_loss_2 = -torch.mean(fake_validity_A) *10
              
                image_bab = generator_4(pred_images_A)
                loss_bab = F.l1_loss(real_B, image_bab) * 20
             
                fsim_value_2 = fsim(denorm(real_B), denorm(image_bab)) * 300
              
                gloss_b2a = g_loss_2 + loss_bab + fsim_value_2

                loss2_history.append(gloss_b2a.item())
                lossG2sum50 = lossG2sum50 + gloss_b2a.item()
           
                gloss_b2a.backward()
                optimizer_G_2.step()
              
                save_image(denorm(pred_images_A), os.path.join(out_dir_2, '{}.png'.format(batches_done)), nrow=5)
                save_image(denorm(real_B), os.path.join(out_dir_4, '{}.png'.format(batches_done)), nrow=5)
                save_image(denorm(image_bab), os.path.join(out_dir_6, '{}.png'.format(batches_done)), nrow=5)
                
                torch.save(critic_2.state_dict(), os.path.join(out_dir, 'critic_2-{}.pt'.format(batches_done)))
                torch.save(generator_2.state_dict(), os.path.join(out_dir, 'generator_2-{}.pt'.format(batches_done)))
                torch.save(generator_4.state_dict(), os.path.join(out_dir_6, 'generator_4-{}.pt'.format(batches_done)))
                print("saved images and state")
    
                generator.load_state_dict(generator_2.state_dict())
              
                np.save(os.path.join(out_dir, 'loss1_history.npy'), loss1_history)
                np.save(os.path.join(out_dir, 'loss2_history.npy'), loss2_history)

               
                print(
                    f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D1 loss: {loss_critic_A.item()}] [D2 loss: {loss_critic_B.item()}] [G1 loss: {gloss_a2b.item()}][G2 loss: {gloss_b2a.item()}] ")
                batches_done += n_critic


        aver_loss_criticA = lossD1sum100 / 100
        aver_loss_criticB = lossD2sum100 / 100

        aver_loss1 = lossG1sum50 / 50
        aver_loss2 = lossG2sum50 / 50


        print(
            f"[Epoch {epoch}/{n_epochs}][D1 loss: {aver_loss_criticA}] [D2 loss: {aver_loss_criticB}] [G1 loss: {aver_loss1}][G2 loss: {aver_loss2}]")

        aver_loss_criticA_history.append(aver_loss_criticA)
        aver_loss_criticB_history.append(aver_loss_criticB)

        aver_loss1_history.append(aver_loss1)
        aver_loss2_history.append(aver_loss2)

        
        np.save(os.path.join(out_dir, 'aver_loss_criticA_history.npy'), aver_loss_criticA_history)
        np.save(os.path.join(out_dir, 'aver_loss_criticB_history.npy'), aver_loss_criticB_history)

        np.save(os.path.join(out_dir, 'aver_loss1_history.npy'), aver_loss1_history)
        np.save(os.path.join(out_dir, 'aver_loss2_history.npy'), aver_loss2_history)

        lossD1sum100 = 0
        lossD2sum100 = 0

        lossG1sum50 = 0
        lossG2sum50 = 0

if __name__ == "__main__":
    train(r"data\Canny_0_shuffle", 32, 12, 5, 10, 0, (1, 32))
