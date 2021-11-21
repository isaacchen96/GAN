import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from self_attention import *


def build_generator(latent_dim):
    inputs = Input(latent_dim, name='inputs')
    dense1 = Dense(7*7*128, name='dense')(inputs)
    reshape = Reshape((7,7,128), name='reshape')(dense1)
    up1    = Conv2DTranspose(64, 3, strides = (2,2), padding='same', activation='relu', name='up1')(reshape)
    bn1    = BatchNormalization(name='bn1')(up1)
    up2    = Conv2DTranspose(32, 3, strides = (2,2), padding='same', activation='relu', name='up2')(bn1)
    bn2    = BatchNormalization(name='bn2')(up2)
    outputs = Conv2D(1, 3, padding='same', activation='tanh', name='output')(bn2)

    generator = Model(inputs, outputs)

    return generator

def build_discriminator(img_shape, use_bn=True):
    inputs = Input(img_shape)
    conv1  = Conv2D(64, 3, strides=(2, 2), padding='same', name='conv1')(inputs)
    if use_bn:
        bn1 = BatchNormalization(name='bn1')(conv1)
        leak1 = LeakyReLU(alpha = 0.3, name='leaky1')(bn1)
    else:
        leak1 = LeakyReLU(alpha=0.3, name='leaky1')(conv1)
    drop1 = Dropout(0.3, name='drop1')(leak1)

    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', name='conv2')(drop1)
    if use_bn:
        bn2 = BatchNormalization(name='bn2')(conv2)
        leak2 = LeakyReLU(alpha = 0.3, name='leaky2')(bn2)
    else:
        leak2 = LeakyReLU(alpha=0.3, name='leaky1')(conv2)
    drop2 = Dropout(0.3, name='drop2')(leak2)

    flat = Flatten(name='flat')(drop2)
    outputs = Dense(1, activation='sigmoid', name='outputs')(flat)

    discriminator = Model(inputs, outputs)

    return discriminator

def build_sa_generator(latent_dim):
    inputs = Input(latent_dim, name='inputs')
    dense1 = Dense(7*7*128, name='dense')(inputs)
    reshape = Reshape((7,7,128), name='reshape')(dense1)
    up1    = Conv2DTranspose(64, 3, strides = (2,2), padding='same', activation='relu', name='up1')(reshape)
    att1 = SelfAttention()(up1)
    bn1    = BatchNormalization(name='bn1')(att1)
    up2    = Conv2DTranspose(32, 3, strides = (2,2), padding='same', activation='relu', name='up2')(bn1)
    att2 = SelfAttention()(up2)
    bn2    = BatchNormalization(name='bn2')(att2)
    outputs = Conv2D(1, 3, padding='same', activation='tanh', name='output')(bn2)

    generator = Model(inputs, outputs)

    return generator

def build_sa_discriminator(img_shape, use_bn=True):
    inputs = Input(img_shape)
    conv1  = Conv2D(64, 3, strides=(2, 2), padding='same', name='conv1')(inputs)
    if use_bn:
        bn1 = BatchNormalization(name='bn1')(conv1)
        leak1 = LeakyReLU(alpha = 0.3, name='leaky1')(bn1)
    else:
        leak1 = LeakyReLU(alpha=0.3, name='leaky1')(conv1)
    drop1 = Dropout(0.3, name='drop1')(leak1)
    att = SelfAttention()(drop1)
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', name='conv2')(att)
    if use_bn:
        bn2 = BatchNormalization(name='bn2')(conv2)
        leak2 = LeakyReLU(alpha = 0.3, name='leaky2')(bn2)
    else:
        leak2 = LeakyReLU(alpha=0.3, name='leaky1')(conv2)
    drop2 = Dropout(0.3, name='drop2')(leak2)
    flat = Flatten(name='flat')(drop2)
    outputs = Dense(1, activation='sigmoid', name='outputs')(flat)

    discriminator = Model(inputs, outputs)

    return discriminator