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
          

class LatentAttention():
    def __init__(self):
        maybe_download_and_extract()
        
        writer = tf.summary.FileWriter('logsnew', graph=tf.get_default_graph())
        self.n_hidden = 500
        self.n_z = 20
        self.batchsize = 10

        self.images = tf.placeholder(tf.float32, [None, 224,224,3])
        z_mean, z_stddev = self.recognition(self.images)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        self.guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(self.guessed_z)
        print('The shape of the generated images are', self.generated_images.shape)
        original_flat = tf.reshape(self.images, [self.batchsize, 224*224*3]) 
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 224*224*3])

        self.generation_loss = (tf.reduce_sum(abs(original_flat - generated_flat),1)) #Added the MSE loss
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        tf.summary.image("Generated Image", self.generated_images, max_outputs=4)
        tf.summary.image("Original Image", self.images, max_outputs=4)
        tf.summary.scalar("GenerationLoss", tf.reduce_mean(self.generation_loss))
        tf.summary.scalar("latentLoss", tf.reduce_mean(self.latent_loss))
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
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

    def train(self):
        dl = dataloader('/home/ksharsh/project_16824/data/dogs/*.jpg',100)
        dlsamples = dl.sampleimages()
        #candd=catsanddogs()
        #samples = candd.getdata()
        # train
        merged_summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('logsnew', graph=sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            samples = sess.run([dlsamples])
            images = samples[0]
            imagesfloat = deepcopy(images)
            """Normalizing the images now"""
            for i in range(images.shape[0]):
                imagesfloat[i] = np.array(images[i], dtype=float) / 255.0 
            coord.request_stop()
            coord.join(threads)
            for epoch in range(10000):
                count = 0
                for idx in range(10):
                    batch = imagesfloat[count*10:(count+1)*10]
                    count = count + 1
                    _, gen_loss, lat_loss, summaries = sess.run((self.optimizer,self.generation_loss, self.latent_loss, merged_summary_op), feed_dict={self.images: batch})
                print('The generator loss is', np.mean(gen_loss))
                print('The latent loss is', np.mean(lat_loss))
               
                writer.add_summary(summaries,epoch)
model = LatentAttention()
model.train()
