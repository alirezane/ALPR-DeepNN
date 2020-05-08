from __future__ import print_function

import os
import pickle
import cv2
import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.Hybrid.golden_threshold import calculate_golden_threshold_matrix
from src.config import hdf5_file_path, image_rows, image_cols, models_dirpath
from src.config import batch_size, workers, multiprocessing, max_queue_size
from src.Hybrid.utils import convert_irregular_output_to_regular
from src.Hybrid.utils import per_digit_accuracy, error_fconfidence_indices
from src.Hybrid.utils import plot_ocr_error, plot_segmentation_prediction
from src.Hybrid.generators import OCRGenerator, SegmentationGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# ---------------------------------------------------------------------------------------- #
# Set model type ['Hybrid', 'OCR']
model_type = 'OCR'
evaluation_number = 1
train_set_name = 'Set2'
validation_set_name = 'Set1'
test_set_name = 'Test'
hdf5_images_name = 'images'
hdf5_labels_name = 'labels'
hdf5_locations_name = 'locations'
hdf5_file_name = 'a_file_paths'
model_number = 4
infer_on_train_data = False
infer_on_validation_data = True
infer_on_test_data = False
# Set Flags
infer_ocr_errors = True
infer_segmentation_errors = False
print_error_images = False
print_false_confidence_images = False
make_list_of_errors_fconfs_passed_npassed_thresh = True
# ---------------------------------------------------------------------------------------- #
# Set model directory
model_dir_path = os.path.join(models_dirpath, 'model-' + str(model_number))
evaluation_dir_path = os.path.join(model_dir_path, 'evaluation-' + str(evaluation_number))
if not os.path.exists(evaluation_dir_path):
    os.makedirs(evaluation_dir_path)
# ---------------------------------------------------------------------------------------- #
# Load Dataset
dataset = h5py.File(hdf5_file_path, mode='r')
# ---------------------------------------------------------------------------------------- #
# Print Number of Data Points
print(dataset[train_set_name + '/' + hdf5_images_name].shape[0], 'train samples')
print(dataset[validation_set_name + '/' + hdf5_images_name].shape[0], 'validation samples')
if infer_on_test_data:
    print(dataset[test_set_name + '/' + hdf5_images_name].shape[0], 'test samples')
# ---------------------------------------------------------------------------------------- #
# Define Data Generators
if infer_ocr_errors:
    ocr_generators = dict()
    ocr_generators['Train'] =  OCRGenerator(
        dataset[train_set_name + '/' + hdf5_images_name],
        dataset[train_set_name + '/' + hdf5_labels_name],
        batch_size
    )
    ocr_generators['Validation'] = OCRGenerator(
        dataset[validation_set_name + '/' + hdf5_images_name],
        dataset[validation_set_name + '/' + hdf5_labels_name],
        batch_size
    )
    if infer_on_test_data:
        ocr_generators['Test'] = OCRGenerator(
            dataset[test_set_name + '/' + hdf5_images_name],
            dataset[test_set_name + '/' + hdf5_labels_name],
            batch_size
        )
if infer_segmentation_errors:
    segmentation_generators = dict()
    segmentation_generators['Train'] = SegmentationGenerator(
        dataset[train_set_name + '/' + hdf5_images_name],
        dataset[train_set_name + '/' + hdf5_locations_name],
        batch_size
    )
    segmentation_generators['Validation'] = SegmentationGenerator(
        dataset[validation_set_name + '/' + hdf5_images_name],
        dataset[validation_set_name + '/' + hdf5_locations_name],
        batch_size
    )
    if infer_on_test_data:
        segmentation_generators['Test'] = SegmentationGenerator(
            dataset[test_set_name + '/' + hdf5_images_name],
            dataset[test_set_name + '/' + hdf5_locations_name],
            batch_size
    )
# ---------------------------------------------------------------------------------------- #
# Load trained model
if model_type == 'Hybrid':
    # Load Hybrid model
    hybrid_predictor_file_path = model_dir_path + "/hybrid_main_predictor"
    json_file = open(hybrid_predictor_file_path + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    hybrid_model = keras.models.model_from_json(loaded_model_json)
    hybrid_model.load_weights(hybrid_predictor_file_path+ ".h5")
    # ---------------------------------------------------------------------------------------- #
    # Load OCR model
    if infer_ocr_errors:
        ocr_predictor_file_path = model_dir_path + "/ocr_main_predictor"
        json_file = open(ocr_predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ocr_model = keras.models.model_from_json(loaded_model_json)
        ocr_layers = [layer.name for layer in ocr_model.layers]
        for layer in ocr_layers:
            ocr_model.get_layer(layer).set_weights(hybrid_model.get_layer(layer).get_weights())
        # ---------------------------------------------------------------------------------------- #
        # Compile Loaded Model
        ocr_model.compile(loss=keras.losses.binary_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
    # ---------------------------------------------------------------------------------------- #
    # Load Segmentation model
    if infer_segmentation_errors:
        seg_predictor_file_path = model_dir_path + "/segmentation_main_predictor"
        json_file = open(seg_predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        seg_model = keras.models.model_from_json(loaded_model_json)
        seg_layers = [layer.name for layer in seg_model.layers]
        for layer in seg_layers:
            seg_model.get_layer(layer).set_weights(hybrid_model.get_layer(layer).get_weights())
        # ---------------------------------------------------------------------------------------- #
        # Compile Loaded Model
        seg_model.compile(loss=keras.losses.binary_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
else:
    ocr_predictor_file_path = model_dir_path + "/ocr_main_predictor"
    json_file = open(ocr_predictor_file_path+ ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    ocr_model = keras.models.model_from_json(loaded_model_json)
    ocr_model.load_weights(ocr_predictor_file_path+ ".h5")
    # ---------------------------------------------------------------------------------------- #
    # Compile Loaded Model
    ocr_model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
print('Models Loaded Successfully')
# -------------------------------Evaluation Function-------------------------------------- #
def evaluate(set_roll, set_name):
    if infer_ocr_errors:
        predictions_file_path = os.path.join(evaluation_dir_path, set_roll)
        predictions_file_name = os.path.join(predictions_file_path, set_name + '_ocr_predictions.pkl')
        if not os.path.isfile(predictions_file_name):
            y_pred_ocr = ocr_model.predict_generator(generator=ocr_generators[set_roll],
                                                     verbose=0,
                                                     workers=workers,
                                                     use_multiprocessing=multiprocessing,
                                                     max_queue_size=max_queue_size)
            if not os.path.exists(predictions_file_path):
                os.makedirs(predictions_file_path)
            with open(predictions_file_name, 'wb') as output:
                pickle.dump(y_pred_ocr, output)
            print('OCR ' + set_roll + ' Predictions made successfully')
        else:
            with open(predictions_file_name, 'rb') as input_file:
                y_pred_ocr = pickle.load(input_file)
            print('OCR ' + set_roll + ' Predictions loaded successfully')
        # ---------------------------------------------------------------------------------------- #
        # Process Predictions
        # Process Validation Predictions
        y_pred_ocr_regular = np.array([convert_irregular_output_to_regular(pred)
                                       for pred in y_pred_ocr])
        y_pred_classes_ocr = np.array([np.argmax(pred, axis=1) for pred in y_pred_ocr_regular])

        y_true_ocr = np.array(dataset[set_name + '/' + hdf5_labels_name])
        y_true_ocr_regular = np.array([convert_irregular_output_to_regular(pred)
                                           for pred in y_true_ocr])
        y_true_classes_ocr = np.array([np.argmax(pred, axis=1) for pred in y_true_ocr_regular])
        # ---------------------------------------------------------------------------------------- #
        # OCR accuracy per digit
        report_dir = os.path.join(evaluation_dir_path, set_roll)
        report_dir = os.path.join(report_dir, 'Per Digit Accuracy')
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        per_digit_accuracy(report_dir, y_pred_classes_ocr, y_true_classes_ocr)
        print('OCR accuracy per digit calculated successfully for ' + set_roll + ' data')
        # --------------------------------------------------------------------------- #
        # Calculate error indices and print them
        # Load or Calculate Golden Threshold matrix
        if os.path.isfile(os.path.join(evaluation_dir_path, 'Confidence_thresholds.pkl')):
            with open(os.path.join(evaluation_dir_path, 'Confidence_thresholds.pkl'), 'rb') as input_file:
                thresholds = pickle.load(input_file)
            print('thresholds loaded from file Confidence_thresholds.pkl successfully')
        else:
            # Predict for Train data
            train_predictions_file_name = os.path.join(evaluation_dir_path, 'Train')
            train_predictions_file_name = os.path.join(train_predictions_file_name,
                                                       train_set_name + '_ocr_predictions.pkl')
            thresholds = calculate_golden_threshold_matrix(ocr_model=ocr_model,
                                                           data_generator=ocr_generators['Train'],
                                                           train_predictions_file=train_predictions_file_name,
                                                           threshold_quantile=0.05)
            with open(evaluation_dir_path + '/Confidence_thresholds.pkl', 'wb') as output:
                pickle.dump(thresholds, output)
            print('thresholds saved to file Confidence_thresholds.pkl')
        # Calculate errors and false confidence indices
        error_indices, false_confidence, passed_thresh, not_passed_thresh = \
            error_fconfidence_indices(y_pred_ocr_regular,
                                      y_pred_classes_ocr,
                                      y_true_classes_ocr,
                                      thresholds)
        print('errors and false confidence indices calculated successfully')
        if make_list_of_errors_fconfs_passed_npassed_thresh:
            # Print false confidence errors
            directory = os.path.join(evaluation_dir_path, set_roll)
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Extract false confidence files list
            f = open(os.path.join(directory, 'false confidences.txt'), 'w')
            for k in false_confidence:
                f.write(str(dataset[set_name + '/' + hdf5_file_name][k][0])[2:-1] + '\n')
            f.close()
            print(set_roll + ' false confidences extracted successfully to file {}'
                  .format(os.path.join(directory, 'false confidences.txt')))

            # Extract errors files list
            f = open(os.path.join(directory, 'errors.txt'), 'w')
            for k in error_indices:
                f.write(str(dataset[set_name + '/' + hdf5_file_name][k][0])[2:-1] + '\n')
            f.close()
            print(set_roll + ' errors extracted successfully to file {}'
                  .format(os.path.join(directory, 'errors.txt')))

            # Extract not passed threshold files list
            f = open(os.path.join(directory, 'passed threshold.txt'), 'w')
            for k in passed_thresh:
                f.write(str(dataset[set_name + '/' + hdf5_file_name][k][0])[2:-1] + '\n')
            f.close()
            print(set_roll + ' passed threshold extracted successfully to file {}'
                  .format(os.path.join(directory, 'passed threshold.txt')))

            # Extract not passed threshold files list
            f = open(os.path.join(directory, 'not passed threshold.txt'), 'w')
            for k in not_passed_thresh:
                f.write(str(dataset[set_name + '/' + hdf5_file_name][k][0])[2:-1] + '\n')
            f.close()
            print(set_roll + ' not passed threshold extracted successfully to file {}'
                  .format(os.path.join(directory, 'not passed threshold.txt')))

        if print_false_confidence_images:
            directory = os.path.join(evaluation_dir_path, set_roll)
            directory = os.path.join(directory, 'False Confidences')
            if not os.path.exists(directory):
                os.makedirs(directory)
            for k in false_confidence:
                fig = plot_ocr_error(y_pred_ocr_regular[k],
                                     y_pred_classes_ocr[k],
                                     y_true_classes_ocr[k],
                                     dataset[set_name + '/' + hdf5_images_name][k],
                                     thresholds)
                plt.tight_layout()
                file_name = str(dataset[set_name + '/' + hdf5_file_name][k][0])[2:-3] + '.png'
                plt.savefig(os.path.join(directory, file_name))
                plt.close()
            print(set_roll + ' false confidences printed successfully')
        # Print errors
        directory = os.path.join(evaluation_dir_path, set_roll)
        directory = os.path.join(directory, 'Errors')
        if print_error_images:
            if not os.path.exists(directory):
                os.makedirs(directory)
            for k in error_indices:
                fig = plot_ocr_error(y_pred_ocr_regular[k],
                                     y_pred_classes_ocr[k],
                                     y_true_classes_ocr[k],
                                     dataset[set_name + '/' + hdf5_images_name][k],
                                     thresholds)
                plt.tight_layout()
                file_name = str(dataset[set_name + '/' + hdf5_file_name][k][0])[2:-3] + '.png'
                plt.savefig(os.path.join(directory, file_name))
                plt.close()
            print(set_roll + ' errors printed successfully')

        # Generate results text file
        result_file = os.path.join(evaluation_dir_path, set_roll)
        result_file = os.path.join(result_file, 'results.txt')
        f = open(result_file, 'w')
        f.write('#' + set_roll + 'data = {}\n'.format(dataset[set_name + '/' + hdf5_images_name].shape[0]))
        f.write('# errors = {} -> {}\n'.format(len(error_indices),
                                               (len(error_indices) / dataset[set_name +
                                                                             '/' +
                                                                             hdf5_images_name].shape[0])))
        f.write('# passed thresh = {} -> {}\n'.format(len(passed_thresh),
                                                      (len(passed_thresh) / dataset[set_name +
                                                                                    '/' +
                                                                                    hdf5_images_name].shape[0])))
        f.write('# not passed thresh = {} -> {}\n'.format(dataset[set_name +
                                                                  '/' +
                                                                  hdf5_images_name].shape[0] - len(passed_thresh),
                                                          (dataset[set_name +
                                                                   '/' +
                                                                   hdf5_images_name].shape[0]
                                                           - len(passed_thresh))
                                                          / dataset[set_name +
                                                                    '/' +
                                                                    hdf5_images_name].shape[0]))
        f.write('# false confidence = {} -> in whole {}\n'.format(len(false_confidence),
                                                                  (len(false_confidence)
                                                                   / dataset[set_name +
                                                                             '/' +
                                                                             hdf5_images_name].shape[0])))
        f.write('# false confidence = {} -> in passed threshold {}\n'.format(len(false_confidence),
                                                                             (len(false_confidence)
                                                                              / len(passed_thresh))))
        f.close()
        print('results file generated successfully')
        # --------------------------------------------------------------------------- #
        # Print Most Important Segmentation Errors
        if model_type == 'Hybrid' and infer_segmentation_errors:
            y_pred_seg = seg_model.predict_generator(generator=segmentation_generators[set_roll],
                                                     verbose=0, workers=0, use_multiprocessing=False)
            # Segmentation Errors
            accs = []
            for error_index in range(dataset[set_name + '/images'].shape[0]):
                if error_index % 1000 == 0 and error_index > 1:
                    print('{}/{}'.format(error_index, dataset[set_name + '/images'].shape[0]))

                box_pred = cv2.resize(y_pred_seg[error_index], (image_cols, image_rows),
                                      interpolation=cv2.INTER_AREA)

                pred = cv2.threshold(box_pred, 0.5, 1, cv2.THRESH_BINARY)[1]
                true = dataset[set_name + '/locations'][error_index]
                intersect = np.logical_and(pred, true)
                intersect = intersect.sum()
                union = np.logical_or(pred, true)
                union = union.sum()
                acc = intersect / union
                accs.append([error_index, acc])

            accs.sort(key=lambda x: x[1])
            error_indices = [x[0] for x in accs if x[1] < 0.75]
            error_values = [x[1] for x in accs if x[1] < 0.75]
            distplot = sns.distplot([x[1] for x in accs])
            fig = distplot.get_figure()
            file_path = os.path.join(evaluation_dir_path, set_roll)
            fig.savefig(os.path.join(file_path, 'segmentation_error_dist.png'))

            for error_index in error_indices:
                seg_plot, seg_plot_heatmap = plot_segmentation_prediction(image=dataset[set_name +
                                                                                        '/images'][error_index],
                                                                          box_pred=y_pred_seg[error_index],
                                                                          box_true=dataset[set_name +
                                                                                           '/locations'][error_index])
                location_errors_directory = os.path.join(evaluation_dir_path, set_roll)
                location_errors_directory = os.path.join(location_errors_directory, 'Locations')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_name = str(dataset[set_name + '/' + hdf5_file_name][k][0])[2:-3] + '.png'
                cv2.imwrite(os.path.join(location_errors_directory, file_name), seg_plot)
                file_name = str(dataset[set_name + '/' + hdf5_file_name][k][0])[2:-3] + '_heatmap.png'
                cv2.imwrite(os.path.join(location_errors_directory, file_name), seg_plot_heatmap)
# -------------------------------Infer on Train------------------------------------------- #
# Predict
# Predict for Train data
if infer_on_train_data:
    evaluate('Train', train_set_name)
# -------------------------------Infer on Validation-------------------------------------- #
# Predict
# Predict for Validation data
if infer_on_validation_data:
    evaluate('Validation', validation_set_name)
# -------------------------------Infer on Test-------------------------------------------- #
# Predict
# Predict for Test data
if infer_on_test_data:
    evaluate('Test', test_set_name)




