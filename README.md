# CifarVAE
Variational Auto Encoder for RGB images. 

You can place the input images in a folder and specify its path in the code. The code lloads the data automatically and runs for 1000 epochs by sampling some images everytime. All these are hyper parameters which can be edited in the code. Also, I have used one encoder and two decoders(one to reconstruct the input image itself and the other to estimate its gradients), you can easily change this to one encoder, one decoder to resemble a traditional VAE.

Also, the cifar data loader is already present in the code itself and you can use that to run VAE on CIFAR images.

I have trained this on imagenet dogs and below are some of the results. The reconstructed image is to the left and the generated image is to the right. We can see that the reconstructed images are blurry. This is the main drawback of the VAE's compared to GAN's. However I still like VAE's mainly because of the nice properties of the underlyinhg latent space. But of late even latent space of GAN's have nice structures and a hybrid architecture combining VAE and GAN's are being used. 

![Alt text](results/genimg0.jpg?raw=true "Reconstructed Image")
![Alt text](results/origimg0.png?raw=true "Original Image")

I have also added the tensorboard support where in the reconstruction loss, latent loss etc get written every epoch and can be visualized using tensorboard.

The data loader for MNIST has been used from https://github.com/Hvass-Labs/TensorFlow-Tutorials .

The overall structure of the code draws inspiration from https://github.com/kvfrans/variational-autoencoder .
