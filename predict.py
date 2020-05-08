from __future__ import print_function

import os
import pickle
import sys
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json

from src.Hybrid.utils import convert_irregular_output_to_regular, plot_ocr_prediction, plot_segmentation_prediction
from src.config import loss_weights

# --------------------------------------------------------------------------- #
# Set prediction mode ['Hybrid', 'Segmentation']
prediction_mode = 'Hybrid'
model_number = 6
model_dir_path = "./Models/Hybrid/model-" + str(model_number)
images_for_prediction_dir = os.path.join(model_dir_path, 'Report/location-test/')
predictions_directory_ocr = os.path.join(model_dir_path, 'Report/location-test/OCR')
predictions_directory_seg = os.path.join(model_dir_path, 'Report/location-test/Detection')
# --------------------------------------------------------------------------- #
# Load model
if prediction_mode == 'Hybrid':
    try:
        predictor_file_path = model_dir_path + "/hybrid_main_predictor"
        json_file = open(predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(predictor_file_path + ".h5")
        model.compile(loss={'OCR': keras.losses.binary_crossentropy,
                            'Segmentation': keras.losses.binary_crossentropy},
                      loss_weights={'OCR': loss_weights[0],
                                    'Segmentation': loss_weights[0]},
                      optimizer=keras.optimizers.Adam(),
                      metrics={'OCR': 'accuracy', 'Segmentation': 'accuracy'})
    except FileNotFoundError:
        print("Hybrid Model weights not found.\n")
        print("Run 'train.py' in 'Hybrid' or 'OCR' mode to train model first.")
        sys.exit()
else:
    try:
        predictor_file_path = model_dir_path + "/hybrid_main_predictor_seg"
        json_file = open(predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(predictor_file_path + ".h5")
    except FileNotFoundError:
        print("Segmentation Model weights not found.\n")
        print("Run 'train.py' in 'Hybrid' mode to train model first.")
        sys.exit()
# --------------------------------------------------------------------------- #
# Load Confidence Threshold Matrix
if prediction_mode == 'Hybrid':
    try:
        with open(os.path.join(model_dir_path, 'Confidence_thresholds.pkl'), 'rb') as input:
            thresholds = pickle.load(input)
    except FileNotFoundError:
        print("Confidence_thresholds.pkl file is missing.\n")
        print("Run 'evaluate.py' in 'Hybrid' or 'OCR' mode and with 'calculate_golden_threshold' flag as"
              " 'True' to create Confidence_thresholds.pkl file")
        sys.exit()
# --------------------------------------------------------------------------- #
# Read Images
images_for_prediction_paths = []
for (dirpath, dirnames, filenames) in os.walk(images_for_prediction_dir):
    images_for_prediction_paths += [os.path.join(dirpath, file) for file in filenames
                                    if os.path.isfile(os.path.join(dirpath, file)) and
                                    file[-3:] in ['jpg']]
images_for_prediction = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in images_for_prediction_paths]
images_for_prediction = [image/255 for image in images_for_prediction]
images_for_prediction = [image.reshape(image.shape[0], image.shape[1], 1) for image in images_for_prediction]
if prediction_mode == 'Hybrid':
    images_for_prediction = [image.reshape(model.input_shape[1:]) for image in images_for_prediction]
images_for_prediction = np.array(images_for_prediction)
# -------------------------------Hybrid (OCR + Segmentation)----------------------------- #
# Make Predictions
if prediction_mode == 'Hybrid':
    predictions = model.predict(np.array(images_for_prediction))
    predictions_ocr = predictions[0]
    predictions_seg = predictions[1]

    predictions_ocr_regular = np.array([convert_irregular_output_to_regular(pred) for pred in predictions_ocr])
    predictions_classes_ocr = np.array([np.argmax(pred, axis=1) for pred in predictions_ocr_regular])
    # --------------------------------------------------------------------------- #
    # Print Predictions OCR
    if not os.path.exists(predictions_directory_ocr):
        os.makedirs(predictions_directory_ocr)
    for k in range(images_for_prediction.shape[0]):
        pred = predictions_classes_ocr[k]
        image = images_for_prediction[k].reshape(model.input_shape[1:3])

        fig, ax = plot_ocr_prediction(predictions_ocr_regular[k],
                                      predictions_classes_ocr[k],
                                      images_for_prediction[k].reshape(model.input_shape[1:3]),
                                      thresholds)
        plt.tight_layout()
        file_name = str(images_for_prediction[k]).split('/')[-1][:-4] + '.png'
        plt.savefig(os.path.join(predictions_directory_ocr, file_name))
        plt.close()
    # --------------------------------------------------------------------------- #
    # Print Predictions Segmentatation
    if not os.path.exists(predictions_directory_seg):
        os.makedirs(predictions_directory_seg)
    for k in range(len(images_for_prediction)):
        seg_plot, seg_plot_heatmap = plot_segmentation_prediction(
            image=images_for_prediction[k].reshape(model.input_shape[1:3]),
            box_pred=predictions_seg[k],
            box_true=False)

        file_name = images_for_prediction[k].split('\\')[-1][:-3] + '.png'
        cv2.imwrite(os.path.join(predictions_directory_seg, file_name), seg_plot)
        file_name = images_for_prediction[k].split('\\')[-1][:-3] + '_heatmap.png'
        cv2.imwrite(os.path.join(predictions_directory_seg, file_name), seg_plot_heatmap)
# -------------------------------Segmentation (Variable sized images)----------------- #
else:
    # Print Predictions Segmentation
    predictions_seg = []
    for img in images_for_prediction:
        predictions_seg.append(model.predict(img.reshape(tuple([1] + list(img.shape)))))
    if not os.path.exists(predictions_directory_seg):
        os.makedirs(predictions_directory_seg)
    for k in range(len(images_for_prediction)):
        seg_plot, seg_plot_heatmap = plot_segmentation_prediction(
            image=images_for_prediction[k].reshape(model.input_shape[1:3]),
            box_pred=predictions_seg[k],
            box_true=False)

        file_name = images_for_prediction[k].split('\\')[-1][:-3] + '.png'
        cv2.imwrite(os.path.join(predictions_directory_seg, file_name), seg_plot)
        file_name = images_for_prediction[k].split('\\')[-1][:-3] + '_heatmap.png'
        cv2.imwrite(os.path.join(predictions_directory_seg, file_name), seg_plot_heatmap)
