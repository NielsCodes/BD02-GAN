import tensorflow as tf
from tensorflow.keras import layers

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tensorflow.keras import layers
import time

from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape

from IPython import display

all_image_path = []

full_image_train_path = "C:/School/MinorAI_dataset/img_align_celeba/img_align_celeba/"
# When running in Local Machine  

# Now from this array 
for path in os.listdir(full_image_train_path):
  if '.jpg' in path:
    all_image_path.append(os.path.join(full_image_train_path, path))
    
image_path_50k = all_image_path[0:500]

print(len(image_path_50k))
# print(image_path_50k)

cropping_box = (30, 55, 150, 175) 

# To load an image from a file, we use the open() function in the Image module, passing it the path to the image.
training_images = [np.array((Image.open(path).crop(cropping_box)).resize((64,64))) for path in image_path_50k]
# print(training_images)

for i in range(len(training_images)):
    training_images[i] = ((training_images[i] - training_images[i].min())/(255 - training_images[i].min()))
    
training_images = np.array(training_images)

noise_shape = 100

#  Generator will upsample our seed using convolutional transpose layers (upsampling layers)
def generator_model():
  generator=Sequential()
  
  # Random noise to 4x4x512 image
  generator.add(Dense(4*4*512, input_shape=[noise_shape]))
  
  #  Next, add a reshape layer to the network to reshape the tensor from the 
  # last layer to a tensor of a shape of (4, 4, 512):
  generator.add(Reshape([4,4,512]))
  generator.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"))
  # BatchNormalization is added to the model after the hidden layer, but before the activation, such as LeakyReLU.
  generator.add(BatchNormalization())
  generator.add(LeakyReLU(alpha=0.2))
  
  generator.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
  generator.add(LeakyReLU(alpha=0.2))
  
  generator.add(BatchNormalization())
  generator.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
  generator.add(LeakyReLU(alpha=0.2))
  generator.add(BatchNormalization())
  generator.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same",
                                  activation='sigmoid'))
  return generator

generator = generator_model()
generator.summary()

def discriminator_model():
  discriminator=Sequential()
  discriminator.add(Conv2D(32, kernel_size=4, strides=2, padding="same",input_shape=[64,64, 3]))
  discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
  discriminator.add(LeakyReLU(0.2)) 
  discriminator.add(BatchNormalization())
  discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(BatchNormalization())
  discriminator.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(Flatten())
  discriminator.add(Dropout(0.5))
  discriminator.add(Dense(1,activation='sigmoid'))
  return discriminator

discriminator = discriminator_model()

discriminator.summary()

GAN =Sequential([generator,discriminator])

discriminator.compile(optimizer='adam',loss='binary_crossentropy')

# When we train this network, we don't want to train the discriminator network, 
# so make it non-trainable before we add it to the adversarial model.
discriminator.trainable = False

GAN.compile(optimizer='adam',loss='binary_crossentropy')

GAN.layers

GAN.summary()

epochs = 300
batch_size = 128

loss_from_discriminator_model=[] # Array to collect loss for the discriminator model

loss_from_generator_model=[] # Array to collect loss for generator model

with tf.device('/gpu:0'):
    for epoch in range(epochs):   # epoch循环
        print(f"Epoch：{epoch+1}")

        
        for i in range(training_images.shape[0]//batch_size):

            if (i)%100 == 0:
                print(f"\tbatch： {i} of {len(training_images)//batch_size}")
            
            noise=np.random.uniform(-1,1,size=[batch_size,noise_shape])
            
            gen_image = generator.predict_on_batch(noise)  
            
            train_dataset = training_images[i*batch_size:(i+1)*batch_size]
            
            train_label=np.ones(shape=(batch_size,1))  
            discriminator.trainable = True  
            
            d_loss1 = discriminator.train_on_batch(train_dataset,train_label)

            
            train_label=np.zeros(shape=(batch_size,1))  
            d_loss2 = discriminator.train_on_batch(gen_image,train_label)

            loss_from_discriminator_model.append(d_loss1+d_loss2)    
            
            
            noise=np.random.uniform(-1,1,size=[batch_size,noise_shape])
            train_label=np.ones(shape=(batch_size,1))  
            discriminator.trainable = False  
            
            
            g_loss = GAN.train_on_batch(noise, train_label)
            
            loss_from_generator_model.append(g_loss)  


        if epoch % 5 == 0: 
            samples = 10
            x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples,100)))

            for k in range(samples):
                plt.subplot(2, 5, k+1)  
                plt.imshow(x_fake[k].reshape(64,64,3))
                plt.xticks([])
                plt.yticks([])


            plt.tight_layout()
            plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        print('Epoch: %d,  Loss: D_real = %.3f, D_fake = %.3f,  G = %.3f' %   (epoch+1, d_loss1, d_loss2, g_loss))

for i in range(5):
    plt.figure(figsize=(7,7))   
    for k in range(20):
            noise=np.random.uniform(-1,1,size=[100,noise_shape])
            new_img=generator.predict(noise)   
            plt.subplot(5, 4, k+1)
            plt.imshow(new_img[k].reshape(64,64,3))  
            plt.xticks([])
            plt.yticks([])
 
    plt.tight_layout()
    plt.savefig('last images')
    plt.show()

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)        





