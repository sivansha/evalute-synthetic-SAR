import cv2
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

# prepare data
model_name = 'sar2dem.1'
path = "./data-hn-xy0n"
save_dir = os.path.join(os.getcwd(), 'saved_models')
patch_size = 32
crop_size = 640
no_patches = -1

if not (crop_size/patch_size).is_integer():
    print("size mismatch")
    exit()
else:
    no_patches = int(pow((crop_size/patch_size),2))

files = []
ref_files = []
dem_files = []
ref_images = []
dem_images = []
train_ref = []
train_dem = []

for (dirpath, dirnames, filenames) in os.walk(path):
    files.extend(filenames)

for file in files:
    if "ProjRef.exr" in file:
        ref_files.append(path + "/" + file)

    if "ProjGradDEM.exr" in file:
        dem_files.append(path + "/" + file)

ref_files.sort()
dem_files.sort()

# print(ref_files)
# print(dem_files)
# print(len(ref_files))
# print(len(dem_files))

start_point = 0
end_point = 1
for i in range(start_point, start_point+end_point):

    print("preparing image " + str(i+1))

    ref_img = cv2.imread(ref_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
    dem_img = cv2.imread(dem_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)

    im_h, im_w, im_d = dem_img.shape
    crop_h_s=int(im_h/2-crop_size/2)
    crop_h_e=int(im_h/2+crop_size/2)
    crop_w_s=int(im_w/2-crop_size/2)
    crop_w_e=int(im_w/2+crop_size/2)

    ref_img = ref_img[crop_h_s:crop_h_e, crop_w_s:crop_w_e]
    dem_img = dem_img[crop_h_s:crop_h_e, crop_w_s:crop_w_e, 0:2]

    for i in range(0, crop_size, patch_size):
        for j in range(0, crop_size, patch_size):

            ref_patch = ref_img[i:i+patch_size, j:j+patch_size]
            dem_patch = dem_img[i:i+patch_size, j:j+patch_size]

            train_ref.append(ref_patch)
            train_dem.append(dem_patch)


no_test = int(len(train_ref) * 0.2)

test_ref = train_ref[0:no_test]
test_ref = np.array(test_ref, dtype=np.float32)
test_ref = test_ref.reshape(test_ref.shape[0], patch_size, patch_size, 1)

train_ref = train_ref[no_test:len(train_ref)]
train_ref = np.array(train_ref, dtype=np.float32)
train_ref = train_ref.reshape(train_ref.shape[0], patch_size, patch_size, 1)
print(train_ref)

test_dem = train_dem[0:no_test]
test_dem = np.array(test_dem, dtype=np.float32)
test_dem = test_dem.reshape(test_dem.shape[0], patch_size, patch_size, 2)

train_dem = train_dem[no_test:len(train_dem)]
train_dem = np.array(train_dem, dtype=np.float32)
train_dem = train_dem.reshape(train_dem.shape[0], patch_size, patch_size, 2)
print(train_dem)

# CNN

model = Sequential()

model.add(Conv2D(8,(5, 5), activation='relu', input_shape=train_ref.shape[1:], padding="SAME"))
model.add(Conv2D(8,(3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(16,(3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(16,(3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(32,(3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(32,(3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(32,(3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(16,(3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(16,(3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(8,(3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(8,(3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(2,(1, 1), activation='relu', padding="SAME"))

# Conv(5x5, 1 -> 8) + relu
# Conv(3x3, 8 -> 8) + relu
# Conv(3x3, 8 -> 16) + relu
# maxpool(2x2)
# Conv(3x3, 16 -> 16) + relu
# Conv(3x3, 16 -> 32) + relu
# maxpool(2x2)
# Conv(3x3, 32 -> 32) + relu
# Conv(3x3, 32 -> 32) + relu
# unpool(2x2)
# Conv(3x3, 32 -> 16) + relu
# Conv(3x3, 16 -> 16) + relu
# unpool(2x2)
# Conv(3x3, 16 -> 8) + relu
# Conv(3x3, 8 -> 8) + relu
# Conv(1x1, 8 -> 2)

# opt = keras.optimizers.rmsprop(lr=0.00001)
opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

model.compile(opt, loss='mse')
model.fit(train_ref, train_dem, epochs=100, batch_size=32, shuffle=False, validation_data=(test_ref, test_dem))

from random import randrange
randNum=randrange(10000)
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_name = model_name + str(randNum)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(test_ref, test_dem, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
