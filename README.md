# Image-to-Image-Translation-using-Conditional-GAN

This is the course project of CSE 803: Computer Vision (Fall 2021) at Michigan State University. 

The goal of this project is to generate colored images from
sketches using a generative model. Conditional GAN-based
architecture is incorporated to accomplish the goal. Basically, this project is motivated by the renowned paper -- [Image-to-image translation with conditional adversarial networks (Isola et al.)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf). The architecture and guidelines suggested by the the paper is incorporated in to this project.  
 
Specifically, this project aims at generating images from
edges/sketches. Given, edge image representation and
the corresponding original image, a conditional GAN model is
trained for generating and discriminating produced/output image. The discriminator is given a
source image and a target image and is asked to decide
whether the target is a realistic transformation of the source
image. Adversarial loss is used to train the generator,
which encourages it to generate believable images in
the target domain. L1 loss between the generated image
and the intended output image is also used to update the
generator. The generator model is encouraged to construct
credible translations of the original image as a result of the
additional loss. Danbooru Sketch Pair: a large collection of 128x128 anime pictures and sketch pair dataset converted from Danbooru2017 is used in this project. In terms of evaluation,
mean absolute error and cross-entropy are used. The underlying U-net-based (with skip connection)
generator has been implemented first. On top of that, the
cGAN architecture has been implemented combining gen-
1erator and discriminator. For more details, see the manuscript: [[PDF]](http://galib19.github.io/files/ImageToImage_2021.pdf).
