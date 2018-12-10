#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "John Hancock"
__copyright__ = "no copyright"
__credits__ = ["GitHub user Ero98"]
__license__ = "no license"
__version__ = "0.1"
__maintainer__ = "John Hancock"
__email__ = "jhancoc4@fau.edu"
__status__ = "Development"
"""This code is based on
    dcgan.py from
    https://github.com/eriklindernoren/Keras-GAN/tree/master/dcgan
    by GitHub user Ero98
    copied December 2nd, 2018
"""
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class ChainGAN():

    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.max_chain_length = 5
        self.gen_0_output_size = 785

        optimizer = Adam(0.0002, 0.5)

        self.build_zeroth_model(optimizer)
        self.build_next_models(optimizer)

    def write_log(self, callback, names, logs, batch_no):
        """
        copied from https://gist.github.com/joelthchao/ef6caa586b647c3c032a4f84d52e3a11
        :param names: list of names of values passed in logs
        :param logs: list of values to plot
        :param batch_no: the training step number
        :return:
        """
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
        pass

    def build_zeroth_model(self, optimizer):
        """
        builds zeroth discriminator and generator
        :return:
        """
        # Build and compile the zeroth discriminator
        self.discriminator_0 = self.build_discriminator_0()
        self.discriminator_0.compile(loss='binary_crossentropy',
                                     optimizer=optimizer,
                                     metrics=['accuracy'])

        # Build the zeroth generator
        self.generator_0 = self.build_generator_0()

        # The zeroth generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator_0(z)

        # For the zeroth combined model we will only train the generator
        self.discriminator_0.trainable = False

        # The zeroth discriminator takes generated images as input and determines validity
        valid = self.discriminator_0(img)

        # The zeroth combined model  (stacked zeroth generator and zeroth discriminator)
        # Trains the zeroth generator to fool the zeroth discriminator
        self.combined_0 = Model(z, valid)
        self.combined_0.compile(loss='binary_crossentropy', optimizer=optimizer)
        pass

    def build_generator_0(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        print("---generator 0 ---")
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator_0(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        print("---discriminator 0 ---")
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_next_models(self, optimizer):
        """
        builds first discriminator and generator
        :return:
        """
        self.chained_models = [self.build_next_model(optimizer) for i in range(self.max_chain_length)]


    def build_next_model(self, optimizer):
        """
        builds first discriminator and generator
        :return:
        """
        # Build and compile the zeroth discriminator
        discriminator = self.build_discriminator_0()
        discriminator.compile(loss='binary_crossentropy',
                                     optimizer=optimizer,
                                     metrics=['accuracy'])

        # Build the first generator
        generator = self.build_generator_1()

        # The first generator takes zeroth generator output and
        # zeroth discriminator output as input and generates imgs
        z = Input(shape=(self.gen_0_output_size,))
        img = generator(z)

        # For the zeroth combined model we will only train the generator
        discriminator.trainable = False

        # The zeroth discriminator takes generated images as input and determines validity
        valid = discriminator(img)

        # The zeroth combined model  (stacked zeroth generator and zeroth discriminator)
        # Trains the zeroth generator to fool the zeroth discriminator
        combined = Model(z, valid)
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        return {'generator': generator, 'discriminator': discriminator, 'combined': combined}

    def build_generator_1(self):

        model = Sequential()
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.gen_0_output_size))
        model.summary()
        model.add(Reshape((7,7,128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        print("--- building ith genertor---")
        model.summary()

        noise = Input(shape=(self.gen_0_output_size,))
        img = model(noise)

        return Model(noise, img)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        log_path = './logs'
        callback = TensorBoard(log_path)
        callback.set_model(self.combined_0)

        for epoch in range(epochs):
            gen_imgs, discriminator_loss_fake = self.train_zeroth_model(X_train, batch_size, callback,
                                                                        epoch, fake, save_interval, valid)
            for i in range(self.max_chain_length):
                gen_imgs, discriminator_loss_fake = self.train_next_model(i, X_train, gen_imgs, discriminator_loss_fake,
                                   batch_size, callback, epoch, fake, save_interval, valid)

    def train_zeroth_model(self, X_train, batch_size, callback, epoch, fake, save_interval, valid):
        """
        Train Zeroth Discriminator
        :param X_train:
        :param batch_size:
        :param callback:
        :param epoch:
        :param fake:
        :param save_interval:
        :param valid:
        :return: generated images, for use with next generator
        """
        # Select a random half of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        gen_imgs = self.generator_0.predict(noise)
        # Train the discriminator (real classified as ones and generated as zeros)
        d_loss_real = self.discriminator_0.train_on_batch(imgs, valid)
        d_loss_fake = self.discriminator_0.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # ---------------------
        #  Train Zeroth Generator
        # ---------------------
        # Train the generator (wants discriminator to mistake images as real)
        g_loss = self.combined_0.train_on_batch(noise, valid)
        # ---------------------
        # Plot the progress
        # and save data for tensorboard
        # ---------------------
        print("%d [D0 loss: %f, acc.: %.2f%%] [G0 loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
        self.write_log(callback, ['Discriminator 0 Loss', 'Discriminator 0 Accuracy',
                                  'Generator 0 Loss'], [d_loss[0], 100 * d_loss[1], g_loss], epoch)
        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            self.save_imgs(epoch, self.generator_0, "mnist_generator_0")
        return gen_imgs, d_loss_fake

    def train_next_model(self, model_num, X_train, gen_prev_output, d0_loss_fake, batch_size, callback, epoch, fake, save_interval,
                         valid):
        # ---------------------
        #  Train first discriminator
        # ---------------------
        # Select a random half of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        # reshape generator 0 output and add loss from previous
        # discriminator
        gen_prev_output = np.reshape(gen_prev_output, (32, self.gen_0_output_size - 1))
        gen_prev_output = np.concatenate((gen_prev_output, np.reshape(32 * [d0_loss_fake[0]], (32, 1))), axis=1)
        gen_imgs = self.chained_models[model_num]['generator'].predict(gen_prev_output)
        # Train the discriminator (real classified as ones and generated as zeros)
        d_loss_real = self.chained_models[model_num]['discriminator'].train_on_batch(imgs, valid)
        d_loss_fake = self.chained_models[model_num]['discriminator'].train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # ---------------------
        #  Train Zeroth Generator
        # ---------------------
        # Train the generator (wants discriminator to mistake images as real)
        g_loss = self.chained_models[model_num]['combined'].train_on_batch(gen_prev_output, valid)
        # ---------------------
        # Plot the progress
        # and save data for tensorboard
        # ---------------------
        model_number = model_num+1
        print("%d [D%d loss: %f, acc.: %.2f%%] [G%d loss: %f]"
              % (epoch, model_number, d_loss[0], 100 * d_loss[1], model_number, g_loss))
        self.write_log(callback, [('Discriminator %d Loss' % model_number), ('Discriminator %d Accuracy' % model_number),
                                  ('Generator %d Loss' % model_number)], [d_loss[0], 100 * d_loss[1], g_loss], epoch)
        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            self.save_imgs(epoch, self.chained_models[model_num]['generator'], ("mnist_generator_%d" % model_number))
        return gen_imgs, d_loss_fake

    def save_imgs(self, epoch, generator, generator_name):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator_0.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s_%d.png" % (generator_name, epoch))
        plt.close()


if __name__ == '__main__':
    dcgan = ChainGAN()
    dcgan.train(epochs=8000, batch_size=32, save_interval=50)
