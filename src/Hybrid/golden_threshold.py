import numpy as np
import pickle
import os

from src.Hybrid.utils import convert_irregular_output_to_regular
from src.utils import digit_vector, char_vector
from src.config import num_digits, workers, multiprocessing, max_queue_size


def calculate_golden_threshold_matrix(ocr_model,
                                      data_generator,
                                      train_predictions_file,
                                      threshold_quantile=0.05):

    if os.path.isfile(train_predictions_file):
        with open(train_predictions_file, 'rb') as input_file:
            y_pred_ocr = pickle.load(input_file)
        print('OCR Training Predictions loaded successfully for calculating predictions')
    else:
        y_pred_ocr = ocr_model.predict_generator(generator=data_generator,
                                             verbose=0,
                                             workers=workers,
                                             use_multiprocessing=multiprocessing,
                                             max_queue_size=max_queue_size)

    # Process Training Predictions
    y_pred_ocr_regular = np.array([convert_irregular_output_to_regular(pred)
                                   for pred in y_pred_ocr])
    y_pred_classes_ocr = np.array([np.argmax(pred, axis=1) for pred in y_pred_ocr_regular])

    thresholds = [[] for i in range(num_digits)]
    quantile = threshold_quantile
    for digit in range(num_digits):
        for digit_class in range(len(digit_vector) + len(char_vector)):
            digit_model_output = (y_pred_classes_ocr[:, digit] - digit_class == 0)
            digit_model_output_indices = [i for i in range(len(y_pred_ocr_regular))
                                          if digit_model_output[i] == True]
            try:
                thresholds[digit].append(np.quantile(
                    y_pred_ocr_regular[digit_model_output_indices, digit, digit_class], quantile))
            except:
                thresholds[digit].append(0)
    thresholds = np.array(thresholds)
    return thresholds
