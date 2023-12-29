# TensorFlow 1 obsolete

# from __future__ import division
# import math
# import json
# import random
# import pprint
# import scipy.misc
# import numpy as np
# from time import gmtime, strftime
# from six.moves import xrange

# import tensorflow as tf
# # import tensorflow.contrib.slim as slim
# import tf_slim as slim

# pp = pprint.PrettyPrinter()

# get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# def show_all_variables():
#   model_vars = tf.trainable_variables()
#   slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# def get_image(image_path, input_height, input_width,
#               resize_height=64, resize_width=64,
#               crop=True, grayscale=False):
#   image = imread(image_path, grayscale)
#   return transform(image, input_height, input_width,
#                    resize_height, resize_width, crop)

# def save_images(images, size, image_path):
#   return imsave(inverse_transform(images), size, image_path)

# def imread(path, grayscale = False):
#   if (grayscale):
#     return scipy.misc.imread(path, flatten = True).astype(np.float)
#   else:
#     return scipy.misc.imread(path).astype(np.float)

# def merge_images(images, size):
#   return inverse_transform(images)

# def merge(images, size):
#   h, w = images.shape[1], images.shape[2]
#   if (images.shape[3] in (3,4)):
#     c = images.shape[3]
#     img = np.zeros((h * size[0], w * size[1], c))
#     for idx, image in enumerate(images):
#       i = idx % size[1]
#       j = idx // size[1]
#       img[j * h:j * h + h, i * w:i * w + w, :] = image
#     return img
#   elif images.shape[3]==1:
#     img = np.zeros((h * size[0], w * size[1]))
#     for idx, image in enumerate(images):
#       i = idx % size[1]
#       j = idx // size[1]
#       img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
#     return img
#   else:
#     raise ValueError('in merge(images,size) images parameter '
#                      'must have dimensions: HxW or HxWx3 or HxWx4')

# def imsave(images, size, path):
#   image = np.squeeze(merge(images, size))
#   return scipy.misc.imsave(path, image)

# def center_crop(x, crop_h, crop_w,
#                 resize_h=64, resize_w=64):
#   if crop_w is None:
#     crop_w = crop_h
#   h, w = x.shape[:2]
#   j = int(round((h - crop_h)/2.))
#   i = int(round((w - crop_w)/2.))
#   return scipy.misc.imresize(
#       x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

# def transform(image, input_height, input_width, 
#               resize_height=64, resize_width=64, crop=True):
#   if crop:
#     cropped_image = center_crop(
#       image, input_height, input_width, 
#       resize_height, resize_width)
#   else:
#     cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
#   return np.array(cropped_image)/127.5 - 1.

# def inverse_transform(images):
#   return (images+1.)/2.

# def to_json(output_path, *layers):
#   with open(output_path, "w") as layer_f:
#     lines = ""
#     for w, b, bn in layers:
#       layer_idx = w.name.split('/')[0].split('h')[1]

#       B = b.eval()

#       if "lin/" in w.name:
#         W = w.eval()
#         depth = W.shape[1]
#       else:
#         W = np.rollaxis(w.eval(), 2, 0)
#         depth = W.shape[0]

#       biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
#       if bn != None:
#         gamma = bn.gamma.eval()
#         beta = bn.beta.eval()

#         gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
#         beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
#       else:
#         gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
#         beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

#       if "lin/" in w.name:
#         fs = []
#         for w in W.T:
#           fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

#         lines += """
#           var layer_%s = {
#             "layer_type": "fc", 
#             "sy": 1, "sx": 1, 
#             "out_sx": 1, "out_sy": 1,
#             "stride": 1, "pad": 0,
#             "out_depth": %s, "in_depth": %s,
#             "biases": %s,
#             "gamma": %s,
#             "beta": %s,
#             "filters": %s
#           };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
#       else:
#         fs = []
#         for w_ in W:
#           fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

#         lines += """
#           var layer_%s = {
#             "layer_type": "deconv", 
#             "sy": 5, "sx": 5,
#             "out_sx": %s, "out_sy": %s,
#             "stride": 2, "pad": 1,
#             "out_depth": %s, "in_depth": %s,
#             "biases": %s,
#             "gamma": %s,
#             "beta": %s,
#             "filters": %s
#           };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
#                W.shape[0], W.shape[3], biases, gamma, beta, fs)
#     layer_f.write(" ".join(lines.replace("'","").split()))

# def make_gif(images, fname, duration=2, true_image=False):
#   import moviepy.editor as mpy

#   def make_frame(t):
#     try:
#       x = images[int(len(images)/duration*t)]
#     except:
#       x = images[-1]

#     if true_image:
#       return x.astype(np.uint8)
#     else:
#       return ((x+1)/2*255).astype(np.uint8)

#   clip = mpy.VideoClip(make_frame, duration=duration)
#   clip.write_gif(fname, fps = len(images) / duration)

# def visualize(sess, dcgan, config, option):
#   image_frame_dim = int(math.ceil(config.batch_size**.5))
#   if option == 0:
#     z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
#     samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
#     save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
#   elif option == 1:
#     values = np.arange(0, 1, 1./config.batch_size)
#     for idx in xrange(dcgan.z_dim):
#       print(" [*] %d" % idx)
#       z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
#       for kdx, z in enumerate(z_sample):
#         z[idx] = values[kdx]

#       if config.dataset == "mnist":
#         y = np.random.choice(10, config.batch_size)
#         y_one_hot = np.zeros((config.batch_size, 10))
#         y_one_hot[np.arange(config.batch_size), y] = 1

#         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
#       else:
#         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

#       save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
#   elif option == 2:
#     values = np.arange(0, 1, 1./config.batch_size)
#     for idx in [random.randint(0, dcgan.z_dim - 1) for _ in xrange(dcgan.z_dim)]:
#       print(" [*] %d" % idx)
#       z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
#       z_sample = np.tile(z, (config.batch_size, 1))
#       #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
#       for kdx, z in enumerate(z_sample):
#         z[idx] = values[kdx]

#       if config.dataset == "mnist":
#         y = np.random.choice(10, config.batch_size)
#         y_one_hot = np.zeros((config.batch_size, 10))
#         y_one_hot[np.arange(config.batch_size), y] = 1

#         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
#       else:
#         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

#       try:
#         make_gif(samples, './samples/test_gif_%s.gif' % (idx))
#       except:
#         save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
#   elif option == 3:
#     values = np.arange(0, 1, 1./config.batch_size)
#     for idx in xrange(dcgan.z_dim):
#       print(" [*] %d" % idx)
#       z_sample = np.zeros([config.batch_size, dcgan.z_dim])
#       for kdx, z in enumerate(z_sample):
#         z[idx] = values[kdx]

#       samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
#       make_gif(samples, './samples/test_gif_%s.gif' % (idx))
#   elif option == 4:
#     image_set = []
#     values = np.arange(0, 1, 1./config.batch_size)

#     for idx in xrange(dcgan.z_dim):
#       print(" [*] %d" % idx)
#       z_sample = np.zeros([config.batch_size, dcgan.z_dim])
#       for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

#       image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
#       make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

#     new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
#         for idx in range(64) + range(63, -1, -1)]
#     make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


# def image_manifold_size(num_images):
#   manifold_h = int(np.floor(np.sqrt(num_images)))
#   manifold_w = int(np.ceil(np.sqrt(num_images)))
#   assert manifold_h * manifold_w == num_images
#   return manifold_h, manifold_w

# ------------------------------------------------------------------------------------------------------------------------------------------ #

# TensorFlow 2 updated code

from __future__ import division
import math
# import json
import random
import pprint
import numpy as np
from time import gmtime, strftime
# from six.moves import xrange

import tensorflow as tf
# import tensorflow_addons as tfa
# from tensorflow.keras.layers import Conv2DTranspose
import moviepy.editor as mpy

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / tf.math.sqrt(k_w * k_h * tf.shape(x)[-1])

def show_all_variables():
    model_vars = tf.compat.v1.trainable_variables()
    for var in model_vars:
        print(var)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale=False):
    image = tf.io.read_file(path)
    if grayscale:
        decoded_image = tf.image.decode_image(image, channels=1)
    else:
        decoded_image = tf.image.decode_image(image)

    # Convert to NumPy array
    numpy_image = tf.make_ndarray(decoded_image)

    # Ensure float32 data type
    return numpy_image.astype(np.float32)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = tf.shape(images)[1], tf.shape(images)[2]
    if tf.shape(images)[3] in (3, 4):
        c = tf.shape(images)[3]
        img = tf.zeros([h * size[0], w * size[1], c], dtype=tf.keras.backend.dtype(images))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif tf.shape(images)[3] == 1:
        img = tf.zeros([h * size[0], w * size[1]], dtype=tf.keras.backend.dtype(images))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('In merge(images, size), the images parameter must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    image_uint8 = (image * 255).astype(np.uint8)
    encoded_image = tf.image.encode_png(image_uint8)
    return tf.io.write_file(path, encoded_image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = tf.shape(x)[:2]
    j = tf.cast(tf.round((h - crop_h) / 2.), dtype=tf.int32)
    i = tf.cast(tf.round((w - crop_w) / 2.), dtype=tf.int32)
    cropped_image = x[j:j + crop_h, i:i + crop_w]
    return tf.image.resize(cropped_image, [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    def _transform(image):
        if crop:
            cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
        else:
            cropped_image = tf.image.resize(image, [resize_height, resize_width])
        return np.array(cropped_image) / 127.5 - 1.

    return tf.py_function(_transform, [image], tf.float32)

def inverse_transform(images):
    return (images + 1.) / 2.

def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = str(w.name).split('/')[0].split('h')[-1]

            B = tf.make_ndarray(b).tolist()

            if "lin/" in w.name:
                W = tf.make_ndarray(w)
                depth = tf.shape(W)[1]
            else:
                W = tf.transpose(w)
                W = tf.make_ndarray(tf.transpose(W, perm=[3, 2, 0, 1]))
                depth = tf.shape(W)[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in B]}
            if bn is not None:
                gamma = tf.make_ndarray(bn.gamma).tolist()
                beta = tf.make_ndarray(bn.beta).tolist()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in gamma]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in beta]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w_t in tf.transpose(W):
                    fs.append({
                        "sy": 1, "sx": 1,
                        "depth": tf.shape(W)[0],
                        "w": [f'%.2f' % elem for elem in tf.make_ndarray(w_t)]
                    })

                lines += f"""
                  var layer_{layer_idx} = {{
                    "layer_type": "fc", 
                    "sy": 1, "sx": 1, 
                    "out_sx": 1, "out_sy": 1,
                    "stride": 1, "pad": 0,
                    "out_depth": {tf.shape(W)[1]}, "in_depth": {tf.shape(W)[0]},
                    "biases": {tf.make_ndarray(biases)},
                    "gamma": {tf.make_ndarray(gamma)},
                    "beta": {tf.make_ndarray(beta)},
                    "filters": {fs}
                  }};"""
            else:
                fs = []
                for w_ in W:
                    fs.append({
                        "sy": 5, "sx": 5,
                        "depth": tf.shape(W)[3],
                        "w": [f'%.2f' % elem for elem in tf.make_ndarray(w_).flatten() if tf.rank(elem) == 0]
                    })

                lines += f"""
                  var layer_{layer_idx} = {{
                    "layer_type": "deconv", 
                    "sy": 5, "sx": 5,
                    "out_sx": {2 ** (int(layer_idx) + 2)}, "out_sy": {2 ** (int(layer_idx) + 2)},
                    "stride": 2, "pad": 1,
                    "out_depth": {tf.shape(W)[0]}, "in_depth": {tf.shape(W)[3]},
                    "biases": {tf.make_ndarray(biases)},
                    "gamma": {tf.make_ndarray(gamma)},
                    "beta": {tf.make_ndarray(beta)},
                    "filters": {fs}
                  }};"""
            layer_f.write(" ".join(lines.replace("'", "").split()))

def make_gif(images, fname, duration=2, true_image=False):
    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return tf.cast(x * 255, dtype=tf.uint8)
        else:
            return tf.cast((x + 1) / 2 * 255, dtype=tf.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)

def visualize(sess, dcgan, config, option):
    image_frame_dim = int(math.ceil(config.batch_size ** 0.5))
    
    if option == 0:
        z_sample = tf.random.uniform(shape=(config.batch_size, dcgan.z_dim), minval=-0.5, maxval=0.5)
        samples = dcgan.sampler(z_sample)
        save_images(samples, [image_frame_dim, image_frame_dim],
                    './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
        
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in range(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = tf.random.uniform(shape=(config.batch_size, dcgan.z_dim), minval=-1, maxval=1)
            
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = tf.one_hot(y, depth=10)
                samples = dcgan.sampler(z_sample, y_one_hot)
            else:
                samples = dcgan.sampler(z_sample)

            save_images(samples, [image_frame_dim, image_frame_dim],
                        './samples/test_arange_%s.png' % (idx))
                        
    elif option == 2:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in [random.randint(0, dcgan.z_dim - 1) for _ in range(dcgan.z_dim)]:
            print(" [*] %d" % idx)
            z = tf.random.uniform(shape=(dcgan.z_dim,), minval=-0.2, maxval=0.2)
            z_sample = tf.tile(tf.expand_dims(z, 0), [config.batch_size, 1])
            
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = tf.one_hot(y, depth=10)
                samples = dcgan.sampler(z_sample, y_one_hot)
            else:
                samples = dcgan.sampler(z_sample)

            try:
                make_gif(samples.numpy(), './samples/test_gif_%s.gif' % (idx))
            except:
                save_images(samples, [image_frame_dim, image_frame_dim],
                            './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
                        
    elif option == 3:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in range(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = tf.zeros([config.batch_size, dcgan.z_dim])
            
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = dcgan.sampler(z_sample)
            make_gif(samples.numpy(), './samples/test_gif_%s.gif' % (idx))
            
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1. / config.batch_size)

        for idx in range(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = tf.zeros([config.batch_size, dcgan.z_dim])
            
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            image_set.append(dcgan.sampler(z_sample).numpy())
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
                         for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w