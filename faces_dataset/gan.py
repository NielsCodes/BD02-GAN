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

# full_image_train_path = '../input/celeba-dataset/img_align_celeba/img_align_celeba'
# When running in Kaggle

full_image_train_path = "C:/School/MinorAI_dataset/img_align_celeba/img_align_celeba/"
# When running in Local Machine  

# Now from this array 
for path in os.listdir(full_image_train_path):
  if '.jpg' in path:
    all_image_path.append(os.path.join(full_image_train_path, path))
    
image_path_50k = all_image_path[0:30000]

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
                                  activation='tanh'))
  return generator

generator = generator_model()
generator.summary()

def discriminator_model():
  discriminator = Sequential()
  discriminator.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=[64,64, 3]))
  discriminator.add(LeakyReLU(alpha=0.2))
  discriminator.add(Dropout(0.4))
  discriminator.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
  discriminator.add(BatchNormalization())
  discriminator.add(LeakyReLU(alpha=0.2))  
  discriminator.add(Dropout(0.4))
  discriminator.add(Flatten())
  discriminator.add(Dense(1, activation='tanh'))
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

epochs = 50
batch_size = 256

loss_from_discriminator_model=[] # Array to collect loss for the discriminator model

loss_from_generator_model=[] # Array to collect loss for generator model

with tf.device('/gpu:0'):
 for epoch in range(epochs):
    print(f"Currently training on Epoch {epoch+1}")
    
    # Loop over each batch in the dataset
    for i in range(training_images.shape[0]//batch_size):
    # Benefits of Double Division Operator over Single Division Operator in Python
    # The Double Division operator in Python returns the floor value for both integer and floating-point arguments after division.
        
        if (i)%100 == 0:
            print(f"\tCurrently training on batch number {i} of {len(training_images)//batch_size}")
        
        #  Start by sampling a batch of noise vectors from a uniform distribution
        # generator receives a random seed as input which is used to produce an image.
        noise=np.random.uniform(-1,1,size=[batch_size, noise_shape])
        
        gen_image = generator.predict_on_batch(noise)
        # We do this by first sampling some random noise from a random uniform distribution, 
        # then getting the generator’s predictions on the noise. 
        # The noise variable is the code equivalent of the variable z, which we discussed earlier.
        
        # Now I am taking real x_train data
        # by sampling a batch of real images from the set of all image
        train_dataset = training_images[i*batch_size:(i+1)*batch_size]
        
        # Create Labels
        # First training on real image
        train_labels_real=np.ones(shape=(batch_size,1))
        
        discriminator.trainable = True
        
        #  Next, train the discriminator network on real images and real labels:
        d_loss_real = discriminator.train_on_batch(train_dataset,train_labels_real)
        
        #Now training on fake image
        train_labels_fake=np.zeros(shape=(batch_size,1))
        
        d_loss_fake = discriminator.train_on_batch(gen_image,train_labels_fake)
        
        # Creating variables to make ready the whole adversarial network
        noise=np.random.uniform(-1,1,size=[batch_size,noise_shape])
        
        # Image Label vector that has all the values equal to 1
        # To fool the Discriminator Network
        train_label_fake_for_gen_training =np.ones(shape=(batch_size,1))
        
        discriminator.trainable = False

        g_loss = GAN.train_on_batch(noise, train_label_fake_for_gen_training)
        
        loss_from_discriminator_model.append(d_loss_real+d_loss_fake)
        
        loss_from_generator_model.append(g_loss)
        
    if epoch % 50 == 0:
        samples = 10
        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples,100)))

        for k in range(samples):
            plt.subplot(2, 5, k+1)
            plt.imshow(x_fake[k].reshape(64,64,3))
            plt.xticks([])
            plt.yticks([])

        
        plt.tight_layout()
        plt.show()
    print('Epoch: %d,  Loss: D_real = %.3f, D_fake = %.3f,  G = %.3f' %   (epoch+1, d_loss_real, d_loss_fake, g_loss))        

print('Training completed with all epochs')

for k in range(20):
          noise=np.random.uniform(-1,1,size=[100,noise_shape])
          im=generator.predict(noise) 
          plt.subplot(5, 4, k+1)
          plt.imshow(im[k].reshape(64,64,3))
          plt.xticks([])
          plt.yticks([])
 
plt.tight_layout()
plt.show()





