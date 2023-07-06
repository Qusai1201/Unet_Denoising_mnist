import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageOps
import os

(train_data, _ ), ( _ , _) = mnist.load_data()

if not os.path.exists('../Data'):
    os.makedirs('../Data')

if not os.path.exists('../Data/training'):
    os.makedirs('../Data/training')

if not os.path.exists('../Data/testing'):
    os.makedirs('../Data/testing')

if not os.path.exists('../Data/validation'):
    os.makedirs('../Data/validation')


train_split = 0.7
test_split = 0.2
validation_split = 0.1

string_len = 10
num_sample = 60000


num_train = int(num_sample * train_split)
num_test = int(num_sample * test_split)
num_val = num_sample - num_train - num_test


def make_image(path , num):

    indices = np.random.randint(len(train_data) ,size=string_len)

    image = train_data[indices[0]]  
    
    for i in range(1 , 10):
        image = np.concatenate((image, train_data[indices[i]]), axis=1)

    img = Image.fromarray(image.astype(np.uint8), "L")
    img = ImageOps.autocontrast(img)
    img.save("../Data/" + path + '/' + str(num) + ".png")


for i in range(num_train):
    make_image('training' , i)

for i in range(num_train, num_train + num_test):
    make_image('testing' , i)

for i in range(num_train + num_test, num_train + num_test + num_val):
    make_image('validation' , i)

print("training Data : " , num_train)
print("testing Data : " , num_test)
print("validation Data : " , str(num_val) + '\n')

print("Dataset generation completed.")
