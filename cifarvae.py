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

class LatentAttention():
    def __init__(self):
        maybe_download_and_extract()
        
        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
        self.n_hidden = 500
        self.n_z = 100
        self.batchsize = 100

        self.images = tf.placeholder(tf.float32, [None, 32,32,3])
        z_mean, z_stddev = self.recognition(self.images)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        self.guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(self.guessed_z)
        print('The shape of the generated images are', self.generated_images.shape)
        original_flat = tf.reshape(self.images, [self.batchsize, 32*32*3]) 
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 32*32*3])

        self.generation_loss = (tf.reduce_sum(abs(original_flat - generated_flat),1)) #Added the MSE loss
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        tf.summary.image("Generated Image", self.generated_images, max_outputs=2)
        tf.summary.scalar("GenerationLoss", tf.reduce_mean(self.generation_loss))
        tf.summary.scalar("latentLoss", tf.reduce_mean(self.latent_loss))
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        tf.summary.scalar("TotalLoss", self.cost)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            input_shapes=(32,32,3)
            inputs = Input(shape=input_shapes)
            model = Sequential()
            c1=Convolution2D(filters=16,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(input_images)
            c2=Convolution2D(filters=32,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(c1)
            f1=Flatten()(c2)
            fc1=Dense(8*8*32, activation='relu')(f1)
            w_mean = Dense(self.n_z)(fc1)
            w_stddev = Dense(self.n_z)(fc1) 
        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            fc1=Dense(500, activation='relu')(z)
            fc2=Dense(8*8*32, activation='relu')(fc1)
            z_matrix = tf.reshape(fc2, [self.batchsize, 8, 8, 32])
            dc1 = Conv2DTranspose(filters=16,kernel_size=(3, 3),strides=2, padding = 'same',activation='relu')(z_matrix)
            dc2 = Conv2DTranspose(filters=3,kernel_size=(3, 3),strides=2, padding = 'same',activation='sigmoid')(dc1)
        return dc2

    def train(self):
        candd=catsanddogs()
        samples = candd.data
        print('The length of the training samples are', len(samples))
        # train
        merged_summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('logs', graph=sess.graph)
            for epoch in range(1000):
                count = 0
                for idx in range(100):
                    batch = samples[count*100:(count+1)*100]
                    count = count + 1
                    _, gen_loss, lat_loss, summaries = sess.run((self.optimizer,self.generation_loss, self.latent_loss, merged_summary_op), feed_dict={self.images: batch})
                    print('The generator loss is', np.mean(gen_loss))
                    print('The latent loss is', np.mean(lat_loss))
               
                writer.add_summary(summaries,epoch*idx)
model = LatentAttention()
model.train()
