import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import PIL.Image
import cv2
import random
from PIL import Image


base_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
# print(base_model.summary())
img_1 = Image.open('mars.jpg')
img_2 = Image.open('eiffel.jpg')
image = Image.blend(img_1,img_2,0.5)
image.save('img_0.jpg')
sample_image = tf.keras.preprocessing.image.load_img('img_0.jpg')
# Image._show(sample_image)
# print(np.shape(sample_image))
# print(type(sample_image))

sample_image1 = tf.keras.preprocessing.image.img_to_array(sample_image)
# print(type(sample_image1))
# print(f'min pix = {sample_image1.min()}, max px= {sample_image1.max()}')

sample_image1 = np.array(sample_image1)/255.0
# print(sample_image1.shape)
# print(f'min pix = {sample_image1.min()}, max px= {sample_image1.max()}')

sample_image1 = tf.expand_dims(sample_image1,axis=0)
# print(np.shape(sample_image1))

# sample_image1 = tf.squeeze(sample_image1)
# # # or np.squeeze
# print(np.shape(sample_image1))
# plt.imshow(sample_image1)
# plt.show()

names = ['mixed3','mixed5','mixed7']
# names = ['mixed8','mixed9']

layers = [base_model.get_layer(name).output for name in names]
deeper_model =  tf.keras.Model(inputs = base_model.input,outputs = layers)
# print(deeper_model.summary())
activations = deeper_model(sample_image1)
# print(activations)
# print(len(activations))
# print(activations.shape)
# x = tf.constant(2.0)

# with tf.GradientTape() as g:
#   g.watch(x)
#   y = x * x * x
# dy_dx = g.gradient(y, x) # Will compute to 12
# x = tf.constant(5.0)
# with tf.GradientTape() as g:
#   g.watch(x)
#   y = x * x * x * x + x * x * x * x *x
# dy_dx = g.gradient(y, x) # Will compute to 3625
# print(dy_dx)


sample_image1 = tf.squeeze(sample_image1,axis=0)
# print(np.shape(sample_image1))

def calc_loss(image,model):
    img_batch = tf.expand_dims(image,axis=0)
    layers_activation  = model(img_batch)
    print(f'activation value = {layers_activation}')
    losses = []

    for act  in layers_activation:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    print('LOSSES (FROM MULTIPLE ACTIVATION LAYERS) = ', losses)
    print('LOSSES SHAPE (FROM MULTIPLE ACTIVATION LAYERS) = ', np.shape(losses))
    print('SUM OF ALL LOSSES (FROM ALL SELECTED LAYERS)= ', tf.reduce_sum(losses))

    return tf.reduce_sum(losses)

# loss = calc_loss(tf.Variable(sample_image1),deeper_model)
# print(loss)

@tf.function
def deepdream(model,image,step_size):
    with tf.GradientTape() as tape :
        tape.watch(image)
        loss = calc_loss(image,model)

    gradians = tape.gradient(loss,image)

    print(f'gradiasn ={gradians}')
    print(f'gradiasn shape ={np.shape(gradians)}')

    gradians  /= tf.math.reduce_std(gradians)

    image = image + gradians*step_size
    image = tf.clip_by_value(image,-1,1)

    return loss , image


def run_deep_drem_model(model,image,steps=100,step_size=0.01):
    image = tf.keras.applications.inception_v3.preprocess_input(image)

    for step in range(steps):
        loss, image = deepdream(model, image, step_size)

        if step % 4000 == 0:
            plt.figure(figsize=(12, 12))
            plt.imshow(deprocess(image))
            plt.show()
            print(f'step {step}, loss{loss}')

    plt.figure(figsize=(12, 12))
    plt.imshow(deprocess(image))
    plt.show()

    return deprocess(image)

def deprocess(image):
    image = 255*(image+1.0)/2.0
    return tf.cast(image,tf.uint8)


# print(sample_image1.shape)

sample_image1 = np.array(tf.keras.preprocessing.image.load_img('img_0.jpg'))
dream_img = run_deep_drem_model(model=deeper_model,image = sample_image1,steps = 4000, step_size = 0.001)
print(dream_img)
dream_img.show()
sample_image1.show()
# print(sample_image1)
# print(dream_img)








