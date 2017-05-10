import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from scipy.misc import imread, imresize
import glob
#from CIFAR.cifarDataLoader import maybe_download_and_extract
#from cifar import catsanddogs
import keras
from keras import layers
from keras.layers import (Activation, Convolution2D,  Dense, Flatten, Input,
                          Permute, Lambda)
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D, ZeroPadding2D, Flatten
import keras.backend as K
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from copy import deepcopy
from network.blocks import buildblocks
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

class sameimageloader():
    def __init__(self, datadir1, datadir2):
        self.datadir1 = datadir1
        self.datadir2 = datadir2
        print('The data dir1 is', datadir1,self.datadir1)
        print('The data dir2 is', datadir2,self.datadir2)

    def returnimages(self,batchsize):
        """This function creates a text file of the images in the folder"""
        images = glob.glob(self.datadir1+"*.jpg")
        imagenames = deepcopy(images)
        for i in range(len(images)):
            impath = images[i]
            imsplit = impath.split('/')
            imagenames[i] = imsplit[len(imsplit)-1]
        if len(imagenames) == 0 :
            print("No jpg images found, please check the paths")
            return
        if batchsize > len(imagenames):
            print("More images requested than present")
            return
    
        imdir1 = imread(self.datadir1+imagenames[0])
        imdir1 = imresize(imdir1,(224,224,3))
        imdir1 = np.array(imdir1, dtype=float) / 255.0
        imdir1 = imdir1[np.newaxis,...]
        imdir2 = imread(self.datadir2+imagenames[0])
        imdir2 = imresize(imdir2,(224,224,3))
        imdir2 = np.array(imdir2, dtype=float) / 255.0
        imdir2 = imdir2[np.newaxis,...]
        for i in range(batchsize):
            if((os.path.exists(self.datadir1 + imagenames[i+1])==False or os.path.exists(self.datadir2 + imagenames[i+1])==False)):
                batchsize = batchsize+1
                print("This image doesn't exist in both the folders", imagenames[i+1])
                continue
            newim1 = imread(self.datadir1 + imagenames[i+1])
            newim2 = imread(self.datadir2 + imagenames[i+1])
            newim1 = imresize(newim1,(224,224,3))
            newim1 = np.array(newim1, dtype=float) / 255.0
            newim1 = newim1[np.newaxis,...]
            newim2 = imresize(newim2,(224,224,3))
            newim2 = np.array(newim2, dtype=float) / 255.0
            newim2 = newim2[np.newaxis,...]
            imdir1 = np.concatenate((imdir1,newim1), axis=0)
            imdir2 = np.concatenate((imdir2,newim2), axis=0)

        return imdir1, imdir2

class blocks():
    def __init__(self):
        pass

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        # Returns
        Output tensor for the block.
        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x


    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        # Returns
        Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

class LatentAttention():
    def __init__(self):
        #maybe_download_and_extract()
        writer = tf.summary.FileWriter('logsnew', graph=tf.get_default_graph())
        K.set_learning_phase(True)
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
        with tf.variable_scope("recognitionold"):
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
 
    def recognitionold(self, input_images):
        with tf.variable_scope("recognition"):
            input_shape = (224,224,3)

            #if input_tensor is None:
            img_input = Input(shape=input_shape)
            #By default the number of axes has to be three
            bn_axis = 3

            #Creating the model now
            block = blocks()
            x = ZeroPadding2D((3, 3))(input_images)
            x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
            x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)

            x = block.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
            x = block.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
            x = block.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

            x = block.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
            x = block.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
            x = block.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
            x = block.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

            x = block.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
            x = block.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
            x = block.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
            x = block.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
            x = block.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
            x = block.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

            x = block.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
            x = block.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
            x = block.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

            x = AveragePooling2D((7, 7), name='avg_pool')(x)
            x = Flatten()(x)
            x = Dense(512, activation='softmax', name='fc200')(x)        
            w_mean = Dense(self.n_z)(x)
            w_stddev = Dense(self.n_z)(x)

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
        imloader = sameimageloader('/home/mscvproject/users/harsha/data/dogs/','/home/mscvproject/users/harsha/data/dogs_grad/')
        imagesfloat, gradimagesfloat = imloader.returnimages(100)
        print('The size of the images 1 are', imagesfloat.shape)
        print('The size of the images 2 are', gradimagesfloat.shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=40)
            writer = tf.summary.FileWriter('logsnew', graph=sess.graph)
            #imagesfloat = self.getimages('/home/mscvproject/users/harsha/data/dogs/*.jpg',100)
            #gradimagesfloat = self.getimages('/home/mscvproject/users/harsha/data/dogs_grad/*.jpg',100)
            testimagesfloat = self.getimages('/home/mscvproject/users/harsha/data/testdogs/*.jpg',10)
            for epoch in range(20000):
                K.set_learning_phase(train)
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
