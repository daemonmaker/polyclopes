import os.path as op
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kk
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from PIL import Image


set_name = 'capsule1'
num_classes = 1
num_bounding_boxes = 1
num_bounding_box_coords = 4
batch_size = 32
epochs = 1

data = np.load('{}_samples.npz'.format(set_name), mmap_mode='r')
samples = data['arr_0']
num_samples, img_width, img_height, img_channels = samples.shape
targets = data['arr_1']
bounding_boxes = data['arr_2']

print('Sample dims: ', samples.shape)
print('Label dims: ', targets.shape)
print('Bounding box dims: ', bounding_boxes.shape)

'''
for idx, (sample, label, bounding_box) in enumerate(zip(samples, targets, bounding_boxes)):
    sample_path = op.join('samples/', '{}.png'.format(idx))
    print(sample_path, '\t', label[0], '\t', bounding_box, '\t', samples[idx].shape, '\t', samples[idx].dtype, '\t', np.min(samples[idx]),
          '\t', np.max(samples[idx]))
    img = Image.fromarray(samples[idx])
    img.save(sample_path)
'''

# Prepare samples
split_percentage = 0.8
num_train_samples = int(split_percentage*num_samples)
num_test_samples = num_samples - num_train_samples
samples = samples.astype('float32')/255.0
train_samples = samples[:num_train_samples]
train_targets = targets[:num_train_samples]
train_bounding_boxes = bounding_boxes[:num_train_samples]
test_samples = samples[num_train_samples:]
test_targets = targets[num_train_samples:]
test_bounding_boxes = bounding_boxes[num_train_samples:]
print("train_samples.shape: ", train_samples.shape)
print("test_samples.shape: ", test_samples.shape)


def make_labels(num_samples, num_classes, targets, num_bounding_boxes, bounding_boxes):
    labels = np.zeros((num_train_samples, num_classes + num_bounding_boxes*num_bounding_box_coords))
    for idx in range(num_samples):
        label_vector = np.zeros((1, num_classes))
        label_vector[targets[idx]] = 1
        labels[idx, :num_classes] = label_vector
        for jdx in range(num_bounding_boxes):
            start_position = jdx*4
            end_position = (jdx+1)*4
            labels[idx, num_classes+start_position:num_classes+end_position] = bounding_boxes[idx, start_position:end_position]
    return labels


train_labels = make_labels(num_train_samples, num_classes, train_targets, num_bounding_boxes, train_bounding_boxes)
test_labels = make_labels(num_test_samples, num_classes, test_targets, num_bounding_boxes, test_bounding_boxes)
print("train_labels.shape: ", train_labels.shape)
print("test_labels.shape: ", test_labels.shape)

# Build Model
input_layer = keras.layers.Input(shape=(img_width, img_height, img_channels))
base_model = MobileNetV2(include_top=False, weights=None)(input_layer)
flattened = keras.layers.Flatten()(base_model)
hidden = keras.layers.Dense(1024)(flattened)
output_layer = keras.layers.Dense(num_classes+num_bounding_boxes*num_bounding_box_coords)(hidden)

model = keras.models.Model(inputs=[input_layer], outputs=[output_layer])
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x=train_samples, y=train_labels, batch_size=batch_size, epochs=epochs)