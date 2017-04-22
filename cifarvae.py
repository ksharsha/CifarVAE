import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from ops import *
from CIFAR.cifarDataLoader import maybe_download_and_extract
from cifar import catsanddogs
import keras
from keras.layers import (Activation, Convolution2D,  Dense, Flatten, Input,
                          Permute, Lambda)
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from copy import deepcopy
"""We will use this class to load images from the directory directly"""
class dataloader():
    def __init__(self, datadir, batch_size):
        self.datadir = datadir
        self.batch_size = batch_size

    def sampleimages(self):
        filename_queue = tf.train.string_input_producer(
             tf.train.match_filenames_once(self.datadir))
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image_orig = tf.image.decode_jpeg(image_file)
        image = tf.image.resize_images(image_orig, [224, 224])
        image.set_shape((224, 224, 3))
        num_preprocess_threads = 1
        min_queue_examples = 150
        images = tf.train.shuffle_batch(
        [image],
        batch_size=self.batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * self.batch_size,
        min_after_dequeue=min_queue_examples)
        return images

class graddataloader():
    def __init__(self, datadir, batch_size):
        self.datadir = datadir
        self.batch_size = batch_size

    def sampleimages(self):
        filename_queue = tf.train.string_input_producer(
             tf.train.match_filenames_once(self.datadir))
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image_orig = tf.image.decode_jpeg(image_file)
        image = tf.image.resize_images(image_orig, [224, 224])
        image.set_shape((224, 224, 3))
        num_preprocess_threads = 1
        min_queue_examples = 150
        images = tf.train.shuffle_batch(
        [image],
        batch_size=self.batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * self.batch_size,
        min_after_dequeue=min_queue_examples)
        return images          

class LatentAttention():
    def __init__(self):
        #maybe_download_and_extract()
        
        writer = tf.summary.FileWriter('logsnew', graph=tf.get_default_graph())
        self.n_hidden = 500
        self.n_z = 512
        self.batchsize = 10

        self.images = tf.placeholder(tf.float32, [None, 224,224,3])
        self.gradients = tf.placeholder(tf.float32, [None, 224,224,3])
        z_mean, z_stddev = self.recognition(self.images)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        self.guessed_z = z_mean + (z_stddev * samples)
        self.random_z = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)

        self.generated_images = self.generation(self.guessed_z)
        self.gradient_images = self.gradient(self.guessed_z)
        print('The shape of the generated images are', self.generated_images.shape)
        original_flat = tf.reshape(self.images, [self.batchsize, 224*224*3]) 
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 224*224*3])
        gradoriginal_flat = tf.reshape(self.gradients, [self.batchsize, 224*224*3])
        gradient_flat = tf.reshape(self.gradient_images, [self.batchsize, 224*224*3])

        self.generation_loss = (tf.reduce_sum(abs(original_flat - generated_flat),1)) #Added the MSE loss
        self.gradient_loss = (tf.reduce_sum(abs(gradoriginal_flat - gradient_flat),1)) #Added the MSE loss
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        tf.summary.image("Generated Image", self.generated_images, max_outputs=4)
        tf.summary.image("Original Image", self.images, max_outputs=4)
        tf.summary.image("Gradient Images", self.gradients, max_outputs=4)
        tf.summary.image("Generated Gradient Images", self.gradient_images, max_outputs=4)
        tf.summary.scalar("GenerationLoss", tf.reduce_mean(self.generation_loss))
        tf.summary.scalar("GradientGenerationLoss", tf.reduce_mean(self.gradient_loss))
        tf.summary.scalar("latentLoss", tf.reduce_mean(self.latent_loss))
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss + self.gradient_loss)
        tf.summary.scalar("TotalLoss", self.cost)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            input_shapes=(224,224,3)
            inputs = Input(shape=input_shapes)
            model = Sequential()
            c1=Convolution2D(filters=16,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(input_images)
            c2=Convolution2D(filters=32,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(c1)
            c3=Convolution2D(filters=64,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(c2)
            c4=Convolution2D(filters=32,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(c3)
            f1=Flatten()(c4)
            fc1=Dense(14*14*32, activation='relu')(f1)
            w_mean = Dense(self.n_z)(fc1)
            w_stddev = Dense(self.n_z)(fc1) 
        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            fc1=Dense(500, activation='relu')(z)
            fc2=Dense(14*14*32, activation='relu')(fc1)
            z_matrix = tf.reshape(fc2, [self.batchsize, 14, 14, 32])
            dc1 = Conv2DTranspose(filters=64,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(z_matrix)
            dc2 = Conv2DTranspose(filters=32,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(dc1)
            dc3 = Conv2DTranspose(filters=16,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(dc2)
            dc4 = Conv2DTranspose(filters=3,kernel_size=(3, 3),strides=2, padding = 'same',activation='sigmoid')(dc3)
        return dc4

    #decoder for the gradient network
    def gradient(self, z):
        with tf.variable_scope("gradientgeneration"):
            fgc1=Dense(500, activation='relu')(z)
            fgc2=Dense(14*14*32, activation='relu')(fgc1)
            zg_matrix = tf.reshape(fgc2, [self.batchsize, 14, 14, 32])
            gc1 = Conv2DTranspose(filters=64,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(zg_matrix)
            gc2 = Conv2DTranspose(filters=32,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(gc1)
            gc3 = Conv2DTranspose(filters=16,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(gc2)
            gc4 = Conv2DTranspose(filters=3,kernel_size=(3, 3),strides=2, padding = 'same',activation='sigmoid')(gc3)
        return gc4

    def getimages(self, direc, num):
        dl = dataloader(direc, num)
        dlsamples = dl.sampleimages()
        with tf.Session() as sessdata:
            sessdata.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            samples = sessdata.run([dlsamples])
            images = samples[0]
            imagesfloat = deepcopy(images)
            """Normalizing the images now"""
            for i in range(images.shape[0]):
                imagesfloat[i] = np.array(images[i], dtype=float) / 255.0
            coord.request_stop()
            coord.join(threads)
       
        return imagesfloat


    def train(self):
        #candd=catsanddogs()
        #samples = candd.getdata()
        train = True
        merged_summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=40)
            writer = tf.summary.FileWriter('logsnew', graph=sess.graph)
            imagesfloat = self.getimages('/home/ksharsh/project_16824/data/dogs/*.jpg',100)
            gradimagesfloat = self.getimages('/home/ksharsh/project_16824/data/dogs_grad/*.jpg',100)
            testimagesfloat = self.getimages('/home/ksharsh/project_16824/data/testdogs/*.jpg',10)
            for epoch in range(20000):
                if train:
                    print("Started Training")
                    count = 0
                    for idx in range(10):
                        batch = imagesfloat[count*10:(count+1)*10]
                        gradbatch = gradimagesfloat[count*10:(count+1)*10]
                        count = count + 1
                        _, grad_loss, gen_loss, lat_loss, summaries = sess.run((self.optimizer,self.gradient_loss,self.generation_loss, self.latent_loss, merged_summary_op), feed_dict={self.images: batch, self.gradients: gradbatch})
                    print('The generator loss is', np.mean(gen_loss))
                    print('The gradient loss is', np.mean(grad_loss))
                    print('The latent loss is', np.mean(lat_loss))
                
                    writer.add_summary(summaries,epoch)
                    if (epoch % 500 == 0):
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        train = False

                else:
                    print("Testing")
                    saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
                    testimgs,testgrads = sess.run((self.generated_images, self.gradient_images), feed_dict={self.images: testimagesfloat})
                    for i in range(10):
                        ims("testimgs/" + str(epoch) + "recimg" + str(i) + ".jpg", testimgs[i])
                        ims("testimgs/" + str(epoch) + "recgradimg" + str(i) + ".jpg", testgrads[i])
                    """Generating from the latent space now"""
                    randz = np.random.normal(0,1,[10,512])
                    genimgs,gengrads = sess.run((self.generated_images, self.gradient_images), feed_dict={self.guessed_z : randz})
                    for i in range(10):
                        ims("testimgs/" + str(epoch) + "genimg" + str(i) + ".jpg", genimgs[i])
                        ims("testimgs/" + str(epoch) + "gengradimg" + str(i) + ".jpg", gengrads[i])

                    train = True

model = LatentAttention()
model.train()
