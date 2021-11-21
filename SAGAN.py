import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
import time
from load_data import *
from build_model import *

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class SAGAN:
    def __init__(self, latent_dim = 100, img_shape=(28,28,1), batch_size = 100, use_bn=True):

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.use_bn = use_bn

        self.train_data, _ = load_mnist()
        self.train_data = \
        tf.data.Dataset.from_tensor_slices(self.train_data).batch(batch_size).shuffle(buffer_size=self.train_data.shape[0])

        self.generator = build_sa_generator(self.latent_dim)
        self.gen_opt = RMSprop()
        self.discriminator = build_sa_discriminator(img_shape=self.img_shape, use_bn=True)
        self.dis_opt = RMSprop()

    def rand_(self, batch_size, latent_dim):
        return tf.random.normal((batch_size, latent_dim))

    def g_train_step(self, batch_size, latent_dim):

        real_label = 0*tf.ones(self.batch_size, 1)
        with tf.GradientTape() as tape:
            gen_img = self.generator(self.rand_(batch_size, latent_dim))
            pred_fake = self.discriminator(gen_img)
            bce_g = BinaryCrossentropy()
            loss_g = bce_g(real_label, pred_fake)
        grads = tape.gradient(loss_g, self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))

        return loss_g

    def d_train_step(self, batch_size, latent_dim, real_img):
        real_label = 0*tf.ones((self.batch_size, 1),tf.float32)
        fake_label = tf.ones((self.batch_size, 1),tf.float32)
        label = tf.concat((real_label, fake_label),axis=0)
        with tf.GradientTape() as tape:
            gen_img = self.generator(self.rand_(batch_size, latent_dim))
            real_img = tf.reshape(real_img,(-1,28,28,1))
            img = tf.concat((real_img, gen_img), axis=0)
            pred = self.discriminator(img)

            bce_d = BinaryCrossentropy()
            loss_d = bce_d(label,pred)
        grads = tape.gradient(loss_d, self.discriminator.trainable_variables)
        self.dis_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        return loss_d


    def train(self, epochs=40, sample_interval=1):

        tr_L_G_avg = []
        tr_L_D_avg = []
        start = time.time()

        for epoch in range(epochs):
            tr_L_G = []
            tr_L_D = []
            for real_img in self.train_data:
                loss_g = self.g_train_step(self.batch_size, self.latent_dim)
                tr_L_G.append(loss_g)
                loss_d = self.d_train_step(self.batch_size, self.latent_dim, real_img)
                tr_L_D.append(loss_d)

            tr_L_G_avg.append(np.mean(tr_L_G))
            tr_L_D_avg.append(np.mean(tr_L_D))
            t_pass = time.time() - start
            m_pass, s_pass = divmod(t_pass, 60)
            h_pass, m_pass = divmod(m_pass, 60)

            if (epoch % sample_interval == 0) or (epoch + 1 == epochs):
                self.sample_images(epoch)
                print('\nTime for pass  {:<4d}: {:<2d} hour {:<3d} min {:<4.3f} sec'.format(epoch + 1, int(h_pass),
                                                                                            int(m_pass), s_pass))
                print('Train Loss Generator     :  {:8.5f}'.format(tr_L_G_avg[-1]))
                print('Train Loss Discriminator :  {:8.5f}'.format(tr_L_D_avg[-1]))
                self.generator.save_weights('C:/Users/Pomelo Chen/Desktop/Python/GANpy/GAN_weights/sagan/sagan_g_{}'.format(epoch+1))
                self.discriminator.save_weights('C:/Users/Pomelo Chen/Desktop/Python/GANpy/GAN_weights/sagan/sagan_d_{}'.format(epoch+1))

        return tr_L_G_avg, tr_L_D_avg


    def sample_images(self, epoch):
        r, c = 5, 5
        gen_imgs = self.generator.predict(self.rand_(25, self.latent_dim))

        # Rescale images 0 - 1
        gen_imgs = 0.5 * (gen_imgs + 1)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig('C:/Users/Pomelo Chen/Desktop/Python/GANpy/SAGAN_pic/%d.png' % (epoch+1))
        plt.close()

sagan = SAGAN()
sagan.generator.summary()
sagan.discriminator.summary()
tr_L_G_avg, tr_L_D_avg = sagan.train(epochs=30, sample_interval=1)
