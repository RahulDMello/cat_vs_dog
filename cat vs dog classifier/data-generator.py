from random import shuffle
import glob
import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage

shuffle_data = True  # shuffle the addresses before saving
hdf5_path = 'train_hdf5/dataset.hdf5'  # address to where you want to save the hdf5 file
cat_dog_train_path = 'train/train/*.jpg'
# read addresses and labels from the 'train' folder
X_train_addr = glob.glob(cat_dog_train_path)
Y_train = [0 if 'cat' in addr else 1 for addr in X_train_addr]  # 0 = Cat, 1 = Dog
# to shuffle data
if shuffle_data:
    c = list(zip(X_train_addr, Y_train))
    shuffle(c)
    addrs, labels = zip(*c)
    
train_shape = (64*64*3, len(X_train_addr))

X_train = np.empty(train_shape)

for i in range(len(X_train_addr)):
    if i%100 == 0 :
        print("image: ", i)
    image = np.array(ndimage.imread(X_train_addr[i], flatten=False))
    my_image = scipy.misc.imresize(image, size=(64, 64, 3)).reshape(64*64*3)
    X_train[:, i] = my_image
    
    
print("saving file ", hdf5_path)
# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("X_train", np.shape(X_train), np.float32)
hdf5_file["X_train"][...] = X_train
hdf5_file.create_dataset("Y_train", np.shape(Y_train), np.int8)
hdf5_file["Y_train"][...] = Y_train

print("file saved")