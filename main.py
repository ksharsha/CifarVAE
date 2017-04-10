import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *
from CIFAR.cifarDataLoader import maybe_download_and_extract

class LatentAttention():
    def __init__(self):
        maybe_download_and_extract()
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.n_hidden = 500
        self.n_z = 20
        self.batchsize = 100

        self.images = tf.placeholder(tf.float32, [None, 784])
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        self.guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(self.guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1) #Right now, looks like this has been written only for binary images.

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2,[self.batchsize, 7*7*32])

            w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        guessedtwos =[]
        guessedones =[]
        guessedzeros = []
        visualization, labels = self.mnist.train.next_batch(self.batchsize)
        gtind = np.argmax(labels, axis=1)
        reshaped_vis = visualization.reshape(self.batchsize,28,28)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    print(len(batch))
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print "epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test, guessedz = sess.run([self.generated_images,self.guessed_z], feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,28,28)
                        #print('The size of guessedz is', (guessedz.shape))
                        #print('The labels are', gtind)
                        for k in range(0,self.batchsize):
                            if gtind[k]==0:
                                guessedzeros.append(guessedz[k])
                            if gtind[k]==1:
                                guessedones.append(guessedz[k])
                            if gtind[k]==2:
                                guessedtwos.append(guessedz[k])
                            
                        ims("results/"+str(epoch)+"_"+str(self.n_z)+".jpg",merge(generated_test[:64],[8,8]))
                        #print('The latent variables of zeros are')
                        #print(guessedzeros)
                        #print('The latent variables of ones are')
                        #print(guessedones)
                        guessedzerox = []
                        guessedzeroy = []
                        guessedonex = []
                        guessedoney = []
                        guessedtwox = []
                        guessedtwoy = []
                        for z in range(0,len(guessedzeros)):
                            guessedzerox.append(guessedzeros[z][0])
                            guessedzeroy.append(guessedzeros[z][1])
                        for z in range(0,len(guessedones)):
                            guessedonex.append(guessedones[z][0])
                            guessedoney.append(guessedones[z][1])
                        for z in range(0,len(guessedtwos)):
                            guessedtwox.append(guessedtwos[z][0])
                            guessedtwoy.append(guessedtwos[z][1])
                        plt.plot(guessedzerox, guessedzeroy, 'ro')
                        plt.plot(guessedonex, guessedoney, 'go')
                        #plt.plot(guessedtwox, guessedtwoy, 'bo')
                        plt.title(str(epoch))
                        plt.savefig("results/"+str(epoch)+"latentplots"+".png")
                        guessedzerox = []
                        guessedzeroy = []
                        guessedonex = []
                        guessedoney = []
                        guessedones =[]
                        guessedzeros = []
                        guessedtwox = []
                        guessedtwoy = []


model = LatentAttention()
model.train()
