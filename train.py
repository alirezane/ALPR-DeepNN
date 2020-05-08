from __future__ import print_function

import h5py
import keras
import tensorflow as tf
import os
import sys

from src.config import hdf5_file_path, image_cols, image_rows
from src.config import loss_weights, batch_size, epochs, workers, multiprocessing, max_queue_size
from src.Hybrid.construct_model import construct_model
from src.Hybrid.utils import make_model_dir, save_model, plot_loss
from src.Hybrid.generators import HybridGenerator, OCRGenerator

# ---------------------------------------------------------------------------------------- #
# Set TF Parameters
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
TF_CONFIG_ = tf.ConfigProto()
TF_CONFIG_.gpu_options.allow_growth = True
sess = tf.Session(config=TF_CONFIG_)
# ---------------------------------------------------------------------------------------- #
# Set training mode ['Hybrid', 'OCR']
training_mode = 'OCR'
train_set_name = 'Set2'
validation_set_name = 'Set1'
test_set_name = 'Test'
hdf5_images_name = 'images'
hdf5_labels_name = 'labels'
hdf5_locations_name = 'locations'
# ---------------------------------------------------------------------------------------- #
# Make model directory
model_dir_path = make_model_dir()
# ---------------------------------------------------------------------------------------- #
# Load DataSet
dataset = h5py.File(hdf5_file_path, mode='r')
# ---------------------------------------------------------------------------------------- #
# Print Number of Data Points
print(dataset[train_set_name + '/' + hdf5_images_name].shape[0], train_set_name + ' samples')
print(dataset[validation_set_name + '/' + hdf5_images_name].shape[0], validation_set_name + ' samples')
# ---------------------------------------------------------------------------------------- #
# Define Data Generators
if training_mode == 'Hybrid':
    my_training_batch_generator = HybridGenerator(
        dataset[train_set_name + '/' + hdf5_images_name],
        dataset[train_set_name + '/' + hdf5_labels_name],
        dataset[train_set_name + '/' + hdf5_locations_name],
        batch_size
        )
    my_validation_batch_generator = HybridGenerator(
        dataset[validation_set_name + '/' + hdf5_images_name],
        dataset[validation_set_name + '/' + hdf5_labels_name],
        dataset[validation_set_name + '/' + hdf5_locations_name],
        batch_size
    )
else:
    my_training_batch_generator = OCRGenerator(
        dataset[train_set_name + '/' + hdf5_images_name],
        dataset[train_set_name + '/' + hdf5_labels_name],
        batch_size
    )
    my_validation_batch_generator = OCRGenerator(
        dataset[validation_set_name + '/' + hdf5_images_name],
        dataset[validation_set_name + '/' + hdf5_labels_name],
        batch_size
    )
# ---------------------------------------------------------------------------------------- #
# Construct and Train Model
# ------------------------------------Hybrid Model---------------------------------------- #
if training_mode == 'Hybrid':
    input_shape = (image_rows, image_cols, 1)
    model = construct_model('Hybrid', input_shape)
    model.compile(loss={'OCR': keras.losses.binary_crossentropy,
                        'Segmentation': keras.losses.binary_crossentropy},
                  loss_weights={'OCR': loss_weights[0],
                                'Segmentation': loss_weights[1]},
                  optimizer=keras.optimizers.Adam(),
                  metrics={'OCR': 'accuracy',
                           'Segmentation': 'accuracy'})

    mc = keras.callbacks.ModelCheckpoint(model_dir_path + '/hybrid_weights{epoch:08d}.h5',
                                         save_weights_only=True, period=5)
    # ---------------------------------------------------------------------------------------- #
    # Fit Model
    try:
        history = model.fit_generator(generator=my_training_batch_generator,
                                      steps_per_epoch=int(dataset[train_set_name +
                                                                  '/' +
                                                                  hdf5_images_name].shape[0] // batch_size),
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=my_validation_batch_generator,
                                      validation_steps=int(dataset[validation_set_name +
                                                                   '/' +
                                                                   hdf5_images_name].shape[0] // batch_size),
                                      workers=workers,
                                      use_multiprocessing=multiprocessing,
                                      max_queue_size=max_queue_size,
                                      callbacks=[mc])
    except KeyboardInterrupt:
        save_model(model=model, model_dir_path=model_dir_path, model_type='Hybrid', save_weights=True)
        print('Model Stopped Training by KeyboardInterrupt and saved to: "{}"'.format(model_dir_path))
        # Make Segmentation and OCR model
        segmentation_model = construct_model('Segmentation')
        ocr_model = construct_model('OCR', input_shape=input_shape)
        # Save Segmentation and OCR model
        save_model(model=segmentation_model, model_dir_path=model_dir_path, model_type='Segmentation',
                   save_weights=False)
        save_model(model=ocr_model, model_dir_path=model_dir_path, model_type='OCR', save_weights=False)
        sys.exit(0)
    # ---------------------------------------------------------------------------------------- #
    # Save Model
    save_model(model=model, model_dir_path=model_dir_path, model_type='Hybrid', save_weights=True)
    # ---------------------------------------------------------------------------------------- #
    # Save Model loss
    plot_loss(history.history['OCR_loss'], history.history['val_OCR_loss'],
              history.history['OCR_acc'], history.history['val_OCR_acc'],
              model_dir_path, 1, 'OCR')
    plot_loss(history.history['Segmentation_loss'], history.history['val_Segmentation_loss'],
              history.history['Segmentation_acc'], history.history['val_Segmentation_acc'],
              model_dir_path, 1, 'Segmentation')
    # ---------------------------------------------------------------------------------------- #
    # Make Segmentation and OCR model
    segmentation_model = construct_model('Segmentation')
    ocr_model = construct_model('OCR', input_shape=input_shape)
    # Save Segmentation Model
    save_model(model=segmentation_model, model_dir_path=model_dir_path, model_type='Segmentation', save_weights=False)
    save_model(model=ocr_model, model_dir_path=model_dir_path, model_type='OCR', save_weights=False)
# ------------------------------------OCR Model------------------------------------------- #
else:
    input_shape = (image_rows, image_cols, 1)
    model = construct_model('OCR', input_shape)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    mc = keras.callbacks.ModelCheckpoint(model_dir_path + '/ocr_weights{epoch:08d}.h5',
                                         save_weights_only=True, period=5)
    # ---------------------------------------------------------------------------------------- #
    # Fit Model
    try:
        history = model.fit_generator(generator=my_training_batch_generator,
                                      steps_per_epoch=int(dataset[train_set_name +
                                                                  '/' +
                                                                  hdf5_images_name].shape[0] // batch_size),
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=my_validation_batch_generator,
                                      validation_steps=int(dataset[validation_set_name +
                                                                   '/' +
                                                                   hdf5_images_name].shape[0] // batch_size),
                                      workers=workers,
                                      use_multiprocessing=multiprocessing,
                                      max_queue_size=max_queue_size,
                                      callbacks=[mc])
    except KeyboardInterrupt:
        save_model(model=model, model_dir_path=model_dir_path, model_type='OCR', save_weights=True)
        print('Model Stopped Training by KeyboardInterrupt and saved to: "{}"'.format(model_dir_path))
        sys.exit(0)
    # ---------------------------------------------------------------------------------------- #
    plot_loss(history.history['loss'], history.history['val_loss'],
              history.history['acc'], history.history['val_acc'],
              model_dir_path, 1, 'OCR')
    # Save Model
    save_model(model=model, model_dir_path=model_dir_path, model_type='OCR')
