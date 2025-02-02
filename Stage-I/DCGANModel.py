# TensorFlow 2 updated code

import os
import time
import math
import glob
import tensorflow as tf
import numpy as np
# from tf.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
# from tf.keras.models import Sequential
from NNArchitectureUtils import *
from ImageTransformation import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
    def __init__(self, input_height=108, input_width=108, crop=True, 
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        self.crop = crop
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.c_dim = c_dim
        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        # batch normalization: deals with poor initialization helps gradient flow
        self.d_bn1 = tf.keras.layers.BatchNormalization(name='d_bn1')
        self.d_bn2 = tf.keras.layers.BatchNormalization(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = tf.keras.layers.BatchNormalization(name='d_bn3')

        self.g_bn0 = tf.keras.layers.BatchNormalization(name='g_bn0')
        self.g_bn1 = tf.keras.layers.BatchNormalization(name='g_bn1')
        self.g_bn2 = tf.keras.layers.BatchNormalization(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = tf.keras.layers.BatchNormalization(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        if self.dataset_name == 'mnist':
            (self.data_X, self.data_y), (_, _) = tf.keras.datasets.mnist.load_data()
            self.c_dim = tf.shape(self.data_X[0])[-1]
        else:
            self.data = glob.glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
            imreadImg = imread(self.data[0])
            if len(imreadImg.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.keras.Input(shape=(self.batch_size, self.y_dim), dtype=tf.float32, name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.keras.Input(shape=[self.batch_size] + image_dims, dtype=tf.float32, name='real_images')
        inputs = self.inputs

        self.z = tf.keras.Input(shape=(None, self.z_dim), dtype=tf.float32, name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = self.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = tf.optimizers.Adam(config.learning_rate, beta_1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.optimizers.Adam(config.learning_rate, beta_1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

        self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        self.writer = tf.summary.create_file_writer("./logs")

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim)).astype(np.float32)

        if config.dataset == 'mnist':
            sample_inputs = self.data_X[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        else:
            sample_files = self.data[0:self.sample_num]
            sample = [get_image(sample_file,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=self.crop,
                                grayscale=self.grayscale) for sample_file in sample_files]
            if self.grayscale:
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
            else:
                self.data = glob.glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
                batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in range(0, batch_idxs):
                if config.dataset == 'mnist':
                    batch_images = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
                else:
                    batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch = [get_image(batch_file,
                                      input_height=self.input_height,
                                      input_width=self.input_width,
                                      resize_height=self.output_height,
                                      resize_width=self.output_width,
                                      crop=self.crop,
                                      grayscale=self.grayscale) for batch_file in batch_files]
                    if self.grayscale:
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                if config.dataset == 'mnist':
                    # Update D network
                    with tf.GradientTape() as tape:
                        d_loss_real = self.d_loss_real(batch_images, batch_labels)
                        d_loss_fake = self.d_loss_fake(self.G(batch_z, batch_labels))
                        d_loss = d_loss_real + d_loss_fake
                    grads = tape.gradient(d_loss, self.d_vars)
                    self.optimizer.apply_gradients(zip(grads, self.d_vars))

                    # Update G network
                    with tf.GradientTape() as tape:
                        g_loss = self.g_loss(self.G(batch_z, batch_labels))
                    grads = tape.gradient(g_loss, self.g_vars)
                    self.optimizer.apply_gradients(zip(grads, self.g_vars))

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    with tf.GradientTape() as tape:
                        g_loss = self.g_loss(self.G(batch_z, batch_labels))
                    grads = tape.gradient(g_loss, self.g_vars)
                    self.optimizer.apply_gradients(zip(grads, self.g_vars))

                    errD_fake = self.d_loss_fake(self.G(batch_z, batch_labels))
                    errD_real = self.d_loss_real(batch_images, batch_labels)
                    errG = self.g_loss(self.G(batch_z, batch_labels))
                else:
                    # Update D network
                    with tf.GradientTape() as tape:
                        d_loss = self.d_loss(batch_images, batch_z)
                    grads = tape.gradient(d_loss, self.d_vars)
                    self.optimizer.apply_gradients(zip(grads, self.d_vars))

                    # Update G network
                    with tf.GradientTape() as tape:
                        g_loss = self.g_loss(self.G(batch_z))
                    grads = tape.gradient(g_loss, self.g_vars)
                    self.optimizer.apply_gradients(zip(grads, self.g_vars))

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    with tf.GradientTape() as tape:
                        g_loss = self.g_loss(self.G(batch_z))
                    grads = tape.gradient(g_loss, self.g_vars)
                    self.optimizer.apply_gradients(zip(grads, self.g_vars))

                    errD_fake = self.d_loss_fake(self.G(batch_z))
                    errD_real = self.d_loss_real(batch_images)
                    errG = self.g_loss(self.G(batch_z))

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (
                    epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }
                        )
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    else:
                        try:
                            samples, d_loss, g_loss = self.sess.run(
                                [self.sampler, self.d_loss, self.g_loss],
                                feed_dict={
                                    self.z: sample_z,
                                    self.inputs: sample_inputs,
                                },
                            )
                            save_images(samples, image_manifold_size(samples.shape[0]),
                                        './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                        except:
                            print("one pic error!...")

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
      with tf.variable_scope("discriminator", reuse=reuse):
          if not self.y_dim:
              h0 = lrelu(tf.keras.layers.Conv2D(self.df_dim, kernel_size=4, strides=2, padding='same', name='d_h0_conv')(image))
              h1 = lrelu(self.d_bn1(tf.keras.layers.Conv2D(self.df_dim*2, kernel_size=4, strides=2, padding='same', name='d_h1_conv')(h0)))
              h2 = lrelu(self.d_bn2(tf.keras.layers.Conv2D(self.df_dim*4, kernel_size=4, strides=2, padding='same', name='d_h2_conv')(h1)))
              h3 = lrelu(self.d_bn3(tf.keras.layers.Conv2D(self.df_dim*8, kernel_size=4, strides=2, padding='same', name='d_h3_conv')(h2)))
              h4 = tf.keras.layers.Flatten()(h3)
              h4 = tf.keras.layers.Dense(1, name='d_h4_lin')(h4)

              return tf.nn.sigmoid(h4), h4
          else:
              yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
              x = conv_cond_concat(image, yb)

              h0 = lrelu(tf.keras.layers.Conv2D(self.c_dim + self.y_dim, kernel_size=4, strides=2, padding='same', name='d_h0_conv')(x))
              h0 = conv_cond_concat(h0, yb)

              h1 = lrelu(self.d_bn1(tf.keras.layers.Conv2D(self.df_dim + self.y_dim, kernel_size=4, strides=2, padding='same', name='d_h1_conv')(h0)))
              h1 = tf.keras.layers.Flatten()(h1)
              h1 = tf.keras.layers.Concatenate()([h1, y])

              h2 = lrelu(self.d_bn2(tf.keras.layers.Dense(self.dfc_dim, name='d_h2_lin')(h1)))
              h2 = tf.keras.layers.Concatenate()([h2, y])

              h3 = tf.keras.layers.Dense(1, name='d_h3_lin')(h2)

              return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
      with tf.variable_scope("generator"):
          if not self.y_dim:
              s_h, s_w = self.output_height, self.output_width
              s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
              s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
              s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
              s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

              # project `z` and reshape
              self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

              self.h0 = tf.keras.layers.Reshape([s_h16, s_w16, self.gf_dim * 8])(self.z_)
              h0 = tf.keras.layers.Activation('relu')(self.g_bn0(self.h0))

              self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
              h1 = tf.keras.layers.Activation('relu')(self.g_bn1(self.h1))

              h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
              h2 = tf.keras.layers.Activation('relu')(self.g_bn2(h2))

              h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
              h3 = tf.keras.layers.Activation('relu')(self.g_bn3(h3))

              h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

              return tf.nn.tanh(h4)
          else:
              s_h, s_w = self.output_height, self.output_width
              s_h2, s_h4 = int(s_h/2), int(s_h/4)
              s_w2, s_w4 = int(s_w/2), int(s_w/4)

              yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
              z = concat([z, y], 1)

              h0 = tf.keras.layers.Activation('relu')(self.g_bn0(tf.keras.layers.Dense(self.gfc_dim, name='g_h0_lin')(z)))
              h0 = tf.keras.layers.Concatenate()([h0, y])

              h1 = tf.keras.layers.Activation('relu')(self.g_bn1(tf.keras.layers.Dense(self.gf_dim*2*s_h4*s_w4, name='g_h1_lin')(h0)))
              h1 = tf.keras.layers.Reshape([s_h4, s_w4, self.gf_dim * 2])(h1)
              h1 = conv_cond_concat(h1, yb)

              h2 = tf.keras.layers.Activation('relu')(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
              h2 = conv_cond_concat(h2, yb)

              return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
      with tf.variable_scope("generator", reuse=True):
          if not self.y_dim:
              s_h, s_w = self.output_height, self.output_width
              s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
              s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
              s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
              s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

              h0 = tf.keras.layers.Reshape([s_h16, s_w16, self.gf_dim * 8])(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'))
              h0 = tf.keras.layers.Activation('relu')(self.g_bn0(h0, train=False))

              h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
              h1 = tf.keras.layers.Activation('relu')(self.g_bn1(h1, train=False))

              h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
              h2 = tf.keras.layers.Activation('relu')(self.g_bn2(h2, train=False))

              h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
              h3 = tf.keras.layers.Activation('relu')(self.g_bn3(h3, train=False))

              h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

              return tf.nn.tanh(h4)
          else:
              s_h, s_w = self.output_height, self.output_width
              s_h2, s_h4 = int(s_h/2), int(s_h/4)
              s_w2, s_w4 = int(s_w/2), int(s_w/4)

              yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
              z = concat([z, y], 1)

              h0 = tf.keras.layers.Activation('relu')(self.g_bn0(tf.keras.layers.Dense(self.gfc_dim, name='g_h0_lin'), train=False))
              h0 = tf.keras.layers.Concatenate()([h0, y])

              h1 = tf.keras.layers.Activation('relu')(self.g_bn1(tf.keras.layers.Dense(self.gf_dim*2*s_h4*s_w4, name='g_h1_lin'), train=False))
              h1 = tf.keras.layers.Reshape([s_h4, s_w4, self.gf_dim * 2])(h1)
              h1 = conv_cond_concat(h1, yb)

              h2 = tf.keras.layers.Activation('relu')(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
              h2 = conv_cond_concat(h2, yb)

              return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
          
    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)

        trX, trY, teX, teY = self.load_data(data_dir)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        X, y = shuffle(X, y, random_state=547)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec

    def load_data(self, data_dir):
        trX = self.load_images(os.path.join(data_dir, 'train-images-idx3-ubyte'), 60000)
        trY = self.load_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'), 60000)
        teX = self.load_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'), 10000)
        teY = self.load_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'), 10000)

        return trX, trY, teX, teY

    def load_images(self, file_path, num_images):
        with open(file_path, 'rb') as fd:
            loaded = np.fromfile(file=fd, dtype=np.uint8)
        return loaded[16:].reshape((num_images, 28, 28, 1)).astype(np.float)

    def load_labels(self, file_path, num_labels):
        with open(file_path, 'rb') as fd:
            loaded = np.fromfile(file=fd, dtype=np.uint8)
        return loaded[8:].reshape((num_labels)).astype(np.float)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0