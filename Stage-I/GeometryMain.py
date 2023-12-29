# TensorFlow 2 updated code

import os
import tensorflow as tf
import numpy as np
from DCGANModel import DCGAN
from ImageTransformation import *

def main(_):
    config = tf.compat.v1.ConfigProto()  # Create a ConfigProto object

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    config.gpu_options.allow_growth = True  # Set GPU memory growth option

    with tf.compat.v1.Session(config=config) as sess:  # Pass the config object
        dcgan = DCGAN(
            input_height=512,
            input_width=512,
            crop=False,
            batch_size=7,
            sample_num=10,
            output_height=512,
            output_width=512,
            dataset_name="data",
            input_fname_pattern="*.png",
            checkpoint_dir="checkpoint",
            sample_dir="output"
        )

        show_all_variables()  # Ensure this function is defined or accessible
        dcgan.train()

    OPTION = 1
    visualize(sess, dcgan, FLAGS, OPTION)  # Assuming FLAGS is defined

if __name__ == '__main__':
    tf.compat.v1.app.run()