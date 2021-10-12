import numpy as np 
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import imageio
import tensorflow as tf
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape

path_celeb = []
train_path_celeb = "C:/School/MinorAI_dataset/img_align_celeba/img_align_celeba/"
for path in os.listdir(train_path_celeb):
    if '.jpg' in path:
        path_celeb.append(os.path.join(train_path_celeb, path))

select_path=path_celeb[0:500]

crop = (30, 55, 150, 175)

images = [np.array((Image.open(path).crop(crop)).resize((64,64))) for path in select_path]
for i in range(len(images)):
    images[i] = ((images[i] - images[i].min())/(255 - images[i].min()))
    
images = np.array(images)

X_train=images

X_train.shape

noise_shape = 100

generator=Sequential()
generator.add(Dense(4*4*512,input_shape=[noise_shape]))
generator.add(Reshape([4,4,512]))
generator.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same",
                                 activation='sigmoid'))

generator.summary()

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

discriminator.summary()

DCGAN =Sequential([generator,discriminator])

discriminator.compile(optimizer='adam',loss='binary_crossentropy')
discriminator.trainable = False

DCGAN.compile(optimizer='adam',loss='binary_crossentropy')

DCGAN.summary()

epochs = 300  
batch_size = 128
D_loss=[]
G_loss=[]

with tf.device('/gpu:0'):
    for epoch in range(epochs):   # epoch
        print(f"Epoch：{epoch+1}")

        # batch
        for i in range(X_train.shape[0]//batch_size):

            if (i)%100 == 0:
                print(f"\tbatch： {i} of {len(X_train)//batch_size}")
            
            noise=np.random.uniform(-1,1,size=[batch_size,noise_shape])
            
            gen_image = generator.predict_on_batch(noise)  
            
            train_dataset = X_train[i*batch_size:(i+1)*batch_size]
            
            train_label=np.ones(shape=(batch_size,1))
            discriminator.trainable = True
            
            d_loss1 = discriminator.train_on_batch(train_dataset,train_label)

            train_label=np.zeros(shape=(batch_size,1))
            d_loss2 = discriminator.train_on_batch(gen_image,train_label)

            D_loss.append(d_loss1+d_loss2)   
            
            noise=np.random.uniform(-1,1,size=[batch_size,noise_shape])
            train_label=np.ones(shape=(batch_size,1)) 
            discriminator.trainable = False 
            
            g_loss = DCGAN.train_on_batch(noise, train_label)
            
            G_loss.append(g_loss)  


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



