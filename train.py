# coding:utf-8

from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

import vae


class ImageDataGenerator_VAE(Sequence):
    def __init__(self, dirpath, target_size, batch_size,
                 featurewise_center=False, samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.0,
                 width_shift_range=0.0, height_shift_range=0.0,
                 brightness_range=None,
                 shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0,
                 fill_mode='nearest', cval=0.0, horizontal_flip=False,
                 vertical_flip=False, rescale=None,
                 preprocessing_function=None, data_format=None,
                 validation_split=0.0):

        image_gen = ImageDataGenerator(featurewise_center, samplewise_center,
                                       featurewise_std_normalization,
                                       samplewise_std_normalization,
                                       zca_whitening, zca_epsilon,
                                       rotation_range, width_shift_range,
                                       height_shift_range, brightness_range,
                                       shear_range, zoom_range,
                                       channel_shift_range, fill_mode, cval,
                                       horizontal_flip, vertical_flip, rescale,
                                       preprocessing_function, data_format,
                                       validation_split)
        self.gen = image_gen.flow_from_directory(dirpath,
                                                 target_size=target_size,
                                                 batch_size=batch_size,
                                                 class_mode='input')

    def __getitem__(self, idx):
        batch_x, batch_y = next(self.gen)
        return batch_x, None

    def __len__(self):
        return self.gen.__len__()


if __name__ == "__main__":
    vae = vae.create_model((140, 140, 3))
    train_geneartor = ImageDataGenerator_VAE("./image",
                                             (140, 140, 3),
                                             16,
                                             rescale=1/255.,
                                             horizontal_flip=True)
    vae.fit_generator(train_geneartor, train_geneartor.__len__(), shuffle=True)
