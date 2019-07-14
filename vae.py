# coding:utf-8

from keras import backend as K
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.models import Model


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def create_model(input_shape, block=6, base_filter=16):
    f = base_filter
    latent_dim = 2

    # encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Conv2D(f, kernel_size=2, strides=2, name='encoder_conv1')(inputs)
    x = BatchNormalization(name='encoder_bn1')(x)
    x = Activation('relu', name='encoder_act1')(x)
    for i in range(block-1):
        f = f * 2
        x = Conv2D(f, kernel_size=2, strides=2,
                   name='encoder_conv'+str(i+2))(inputs)
        x = BatchNormalization(name='encoder_bn'+str(i+2))(x)
        x = Activation('relu', name='encoder_act'+str(i+2))(x)
    x = Flatten(name='encoder_flatten')(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,),
               name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(2*2, name='decoder_dense1')(latent_inputs)
    x = BatchNormalization(name='decoder_bn1')(x)
    x = Activation('relu', name='decoder_act1')(x)
    x = Reshape((2, 2, 1), name='decoder_reshape1')(x)
    for i in range(block):
        x = Conv2DTranspose(f, kernel_size=2, strides=2,
                            padding='same', name='decoder_convT'+str(i+1))(x)
        x = BatchNormalization(name='decoder_bn'+str(i+2))(x)
        x = Activation('relu', name='decoder_bn'+str(i+2))(x)
        f = f//2

    x1 = Conv2DTranspose(
        input_shape[2], kernel_size=4, padding='same',
        name='decoder_convT'+str(block))(x)
    x1 = BatchNormalization()(x1)
    out1 = Activation('sigmoid')(x1)

    x2 = Conv2DTranspose(1, kernel_size=4, padding='same')(x)
    x2 = BatchNormalization()(x2)
    out2 = Activation('sigmoid')(x2)
    decoder = Model(latent_inputs, [out1, out2], name='decoder')
    outputs_mu, outputs_sigma_2 = decoder(encoder(inputs)[2])
    vae = Model(inputs, [outputs_mu, outputs_sigma_2], name='vae')

    # loss
    m_vae_loss = (K.flatten(inputs) - K.flatten(outputs_mu)
                  )**2 / K.flatten(outputs_sigma_2)
    m_vae_loss = 0.5 * K.sum(m_vae_loss)

    a_vae_loss = K.log(2 * 3.14 * K.flatten(outputs_sigma_2))
    a_vae_loss = 0.5 * K.sum(a_vae_loss)

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(kl_loss + m_vae_loss + a_vae_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae
