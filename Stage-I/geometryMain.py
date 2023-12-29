# TensorFlow 1 obsolete

# import os
# import tensorflow as tf
# import scipy.misc
# import numpy as np
# from geometryModel import DCGAN
# from utils import pp, visualize, to_json, show_all_variables

# def main(_):
#   run_config = tf.ConfigProto()
#   print("run_config??",run_config)
#   run_config.gpu_options.allow_growth=True

#   with tf.Session(config=run_config) as sess:
#     dcgan = DCGAN(
#         sess,
#         input_width=512,
#         input_height=512,
#         output_width=512,
#         output_height=512,
#         batch_size=7,
#         sample_num=10,
#         dataset_name="data",
#         input_fname_pattern="*.png",
#         crop=False,
#         checkpoint_dir="checkpoint",
#         sample_dir="output")
    
#     show_all_variables()
#     dcgan.train()
      

# OPTION = 1
# visualize(sess, dcgan, FLAGS, OPTION)

# if __name__ == '__main__':
#   tf.app.run()

# ------------------------------------------------------------------------------------------------------------------------------------------ #

# TensorFlow 2 updated code

import os
import tensorflow as tf
import numpy as np
from DCGANModel import DCGAN
from ImageTransformation import *

def main(_):
    run_config = tf.config.experimental.list_physical_devices('GPU')
    for gpu in run_config:
        tf.config.experimental.set_memory_growth(gpu, True)

    with tf.compat.v1.Session(config=run_config) as sess:
        dcgan = DCGAN(
            sess,
            input_width=512,
            input_height=512,
            output_width=512,
            output_height=512,
            batch_size=7,
            sample_num=10,
            dataset_name="data",
            input_fname_pattern="*.png",
            crop=False,
            checkpoint_dir="checkpoint",
            sample_dir="output")

        show_all_variables()
        dcgan.train()

    OPTION = 1
    visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.compat.v1.app.run()