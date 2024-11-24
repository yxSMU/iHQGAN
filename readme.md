

##### iHQGAN: A Lightweight Invertible Hybrid Quantum-Classical Generative Adversarial Network for Unsupervised Image-to-Image Translation,2024

------

<sub>This repository is the implementation of</span>  [iHQGAN: A Lightweight Invertible Hybrid Quantum-Classical Generative Adversarial Network for Unsupervised Image-to-Image Translation](https://arxiv.org/abs/2411.13920)</sub>

<sub>Leveraging quantum computing’s intrinsic properties to enhance machine learning has shown promise, with quantum generative adversarial networks (QGANs) demonstrating benefits in data generation. However, the application of QGANs to complex unsupervised image-to-image (I2I) translation remains unexplored. Moreover, classical neural networks often suffer from large parameter spaces, posing challenges for GAN-based I2I methods. Inspired by the fact that unsupervised I2I translation is essentially an approximate reversible problem, we propose a lightweight invertible hybrid quantum-classical unsupervised I2I translation model — iHQGAN, by harnessing the invertibility of quantum computing. Specifically, iHQGAN employs two mutually approximately reversible quantum generators with shared parameters, effectively reducing the parameter scale. To ensure content consistency between generated and source images, each quantum generator is paired with an assisted classical neural network (ACNN), enforcing a unidirectional cycle consistency constraint between them. Simulation experiments were conducted on 19 sub-datasets across three tasks. Qualitative and quantitative assessments indicate that iHQGAN effectively performs unsupervised I2I translation with excellent generalization and can outperform classical methods that use low-complexity CNN-based generators. Additionally, iHQGAN, as with classical reversible methods, reduces the parameter scale of classical irreversible methods via a reversible mechanism. This study presents the first versatile quantum solution for unsupervised I2I translation, extending QGAN research to more complex image generation scenarios and offering a quantum approach to decrease the parameters of GAN-based unsupervised I2I translation methods. </sub>

[Paper](https://arxiv.org/abs/2411.13920)

###### The framework of iHQGAN

------

<sub>The overall architecture of iHQGAN. iHQGAN consists of two quantum generators, two assisted classical networks (ACNNs), and two classical critics. The quantum generators $G$ and $F$ comprise $p$ sub-quantum generators. ACNNs are utilized to achieve consistency loss, while the critics implement adversarial loss.During the alternating training of the two quantum generators, their respective parameters $\theta_G$ and $\omega_F$ are interchangeably assigned to facilitate parameter sharing.</sub>

![iHQGAN_framework](https://github.com/yxSMU/iHQGAN/raw/main/Fig/iHQGAN_framework.png)



###### Experiment Results

------

<sub>unsupervised Image-to-Image Translation task:  $ Edge Detection$</sub>

<img src="https://github.com/yxSMU/iHQGAN/raw/main/Fig/Edge_Dection.png" alt="Edge_Dection.png" width="460">

<sub>unsupervised Image-to-Image Translation task:  $ Font$  $Style$  $Transfer$</sub>

<img src="https://github.com/yxSMU/iHQGAN/raw/main/Fig/Font_Style_Transfer.png" alt="Font_Style_Transfe" width="460">

<sub>unsupervised Image-to-Image Translation task:   $Image$ $Denosing$ </sub>

<img src="https://github.com/yxSMU/iHQGAN/raw/main/Fig/Image_Denoising.png" alt="Image Denoising" width="380">

<p> <sub>Leveraging quantum computing’s intrinsic . </sub> </p>