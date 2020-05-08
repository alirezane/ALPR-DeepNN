import itertools
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from shutil import copyfile
from matplotlib import transforms
from sklearn.metrics import confusion_matrix

from src.config import models_dirpath, num_digits, image_rows, image_cols
from src.utils import char_vector, digit_vector, digit_char_vector



def make_model_dir():
    directory_created = False
    model_number = 1
    while not directory_created:
        model_dir_path = models_dirpath + "/model-" + str(model_number)
        if os.path.exists(model_dir_path):
            model_number = model_number + 1
        else:
            os.makedirs(model_dir_path)
            directory_created = True
    copyfile('src/config.py', os.path.join(model_dir_path, 'config.py'))
    copyfile('src/Hybrid/construct_model.py', os.path.join(model_dir_path, 'construct_model.py'))
    # copyfile('src/Hybrid/utils.py', os.path.join(model_dir_path, 'utils.py'))
    return model_dir_path


def plot_loss(train_loss, val_loss, train_acc, val_acc, model_dir_path, start_epoch, type):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.arange(start_epoch, len(train_loss) + 1),
               train_loss[start_epoch - 1:], color='b',
               label=type + " Training loss")
    ax[0].plot(np.arange(start_epoch, len(val_loss) + 1),
               val_loss[start_epoch - 1:], color='r',
               label=type + " Validation loss", axes =ax[0])
    ax[0].legend(loc='best', shadow=True)

    ax[1].plot(np.arange(start_epoch, len(train_acc) + 1),
               train_acc[start_epoch - 1:], color='b',
               label=type + " Training accuracy")
    ax[1].plot(np.arange(start_epoch, len(val_acc) + 1),
               val_acc[start_epoch - 1:], color='r',
               label=type + " Validation accuracy")
    ax[1].legend(loc='best', shadow=True)
    plt.savefig(os.path.join(model_dir_path, type + '_loss.png'))
    plt.close()


def rainbow_text(x, y, strings, colors, orientation='horizontal',
                 ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        if orientation == 'horizontal':
            t = transforms.offset_copy(
                text.get_transform(), x=ex.width, units='dots')
        else:
            t = transforms.offset_copy(
                text.get_transform(), y=ex.height, units='dots')


def convert_irregular_output_to_regular(output):
    regular = np.block([
        [output[:len(digit_vector)], np.zeros(len(char_vector))],
        [output[len(digit_vector):2*len(digit_vector)], np.zeros(len(char_vector))],
        [np.zeros(len(digit_vector)), output[2*len(digit_vector):2*len(digit_vector)+len(char_vector)]],
        [output[2 * len(digit_vector) + len(char_vector):3 * len(digit_vector) + len(char_vector)],
         np.zeros(len(char_vector))],
        [output[3 * len(digit_vector) + len(char_vector):4 * len(digit_vector) + len(char_vector)],
         np.zeros(len(char_vector))],
        [output[4 * len(digit_vector) + len(char_vector):5 * len(digit_vector) + len(char_vector)],
         np.zeros(len(char_vector))],
        [output[5 * len(digit_vector) + len(char_vector):6 * len(digit_vector) + len(char_vector)],
         np.zeros(len(char_vector))],
        [output[6 * len(digit_vector) + len(char_vector):7 * len(digit_vector) + len(char_vector)],
         np.zeros(len(char_vector))],
    ])
    return regular


def save_model(model, model_dir_path, model_type, save_weights=True):
    model_json = model.to_json()
    if model_type == 'Hybrid':
        json_file_name = 'hybrid_main_predictor.json'
        weights_file_name = 'hybrid_main_predictor.h5'
    elif model_type == 'OCR':
        json_file_name = 'ocr_main_predictor.json'
        weights_file_name = 'ocr_main_predictor.h5'
    elif model_type == 'Segmentation':
        json_file_name = 'segmentation_main_predictor_seg.json'
        weights_file_name = 'segmentation_main_predictor_seg.h5'
    predictor_file_path = os.path.join(model_dir_path, json_file_name)
    with open(predictor_file_path, "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    print('\nModel saved to json file {}\n'.format(json_file_name))
    if save_weights:
        model.save_weights(os.path.join(model_dir_path, weights_file_name))
        print('\nModel weights saved to file {}\n'.format(weights_file_name))


def per_digit_accuracy(report_dir, y_pred_classes, y_true_classes):
    f = open(os.path.join(report_dir, 'PerDigitAcc.txt'), 'w')
    pred = y_pred_classes.flatten()
    true = y_true_classes.flatten()
    acc = (pred == true).sum() / (pred == true).shape[0]
    f.write('Overall Accuracy in OCR (#Correctly read '
            'characters/#Read characters): {}%\n'.format(acc * 100))
    for i in range(num_digits):
        pred = y_pred_classes[:, i]
        true = y_true_classes[:, i]
        acc = (pred == true).sum() / (pred == true).shape[0]
        confusion_mtx = confusion_matrix(true, pred)
        plt.figure(figsize=(15, 10))
        plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(set(true)))
        plt.xticks(tick_marks, [digit_char_vector[x] for x in set(true)], rotation=45)
        plt.yticks(tick_marks, [digit_char_vector[x] for x in set(true)])

        thresh = confusion_mtx.max() / 2.
        for j, k in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
            plt.text(k, j, confusion_mtx[j, k],
                     horizontalalignment="center",
                     color="white" if confusion_mtx[j, k] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Digit {} Accuracy : {}%'.format(i + 1, round(acc * 100, 5)))
        plt.tight_layout()
        file_name = 'Digit_' + str(i + 1) + '_Confusion Matrix.png'
        plt.savefig(os.path.join(report_dir, file_name))
        plt.close()
        f.write('Accuracy in Digit {}: {}\n'.format(i + 1, round(acc * 100, 5)))
    f.close()


def error_fconfidence_indices(y_pred, y_pred_classes, y_true_classes, thresholds):
    passed_thresh = [i for i in range(y_pred.shape[0]) if pass_thresh(y_pred[i], thresholds)]
    not_passed_thresh = [i for i in range(y_pred.shape[0]) if not pass_thresh(y_pred[i], thresholds)]
    true_pred = (y_pred_classes - y_true_classes == 0)
    true_pred = [all(pred) for pred in true_pred]
    error_indices = [i for i in range(len(true_pred)) if true_pred[i] is False]
    false_confidence_indices = [x for x in error_indices if x in passed_thresh]
    return error_indices, false_confidence_indices, passed_thresh, not_passed_thresh


def plot_ocr_error(error_pred, error_pred_classes, error_true_classes, error_image, thresholds):
    error_image = error_image.reshape(image_rows, image_cols)
    error_image = np.float32(error_image)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(error_image, cmap='gray')
    pred_colors = ['black']
    true_colors = ['black']
    confidence_colors = ['black']
    pred = [digit_char_vector[j] for j in error_pred_classes]
    true = [digit_char_vector[j] for j in error_true_classes]
    for i in range(num_digits):
        if pred[i] == true[i]:
            pred_colors.append('green')
        else:
            pred_colors.append('red')
        true_colors.append('black')
    confidence = ['Conf.'] + ['{:.5f}'.format(round(max(x), 5)) for x in error_pred]
    for i in range(num_digits):
        if error_pred[i, error_pred_classes[i]] < thresholds[i, error_pred_classes[i]]:
            confidence_colors.append('red')
        else:
            confidence_colors.append('green')

    spacing = 45
    pred = ['Pred:'] + pred
    true = ['True'] + true
    for i in range(len(true)):
        rainbow_text(-10 + spacing * i, -45, [true[i]], [true_colors[i]], size=25)
    for i in range(len(pred)):
        rainbow_text(-10 + spacing * i, -30, [pred[i]], [pred_colors[i]], size=25)
    for i in range(len(confidence)):
        rainbow_text(-10 + spacing * i, -15, [confidence[i]], [confidence_colors[i]], size=15)
    return fig


def plot_ocr_prediction(prediction, prediction_classes, image, thresholds):
    image = image.reshape(image_rows, image_cols)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    colors = ['black']
    confidence_colors = ['black']
    pred = [digit_char_vector[j] for j in prediction_classes]
    pred = [x + '        ' if len(x) == 3 else x + '     ' for x in pred]
    for i in range(num_digits):
            colors.append('black')
    confidence = ['{:.5f}'.format(round(max(x), 5)) for x in pred]
    confidence = ['|' + x + '|' for x in confidence]
    for i in range(num_digits):
        if prediction[i, prediction_classes[i]] < thresholds[i, prediction_classes[i]]:
            confidence_colors.append('red')
        else:
            confidence_colors.append('green')
    rainbow_text(-10, -30, ['Pred: '] + pred, colors, size=18)
    rainbow_text(-10, -10, ['Conf.: '] + confidence, confidence_colors, size=10)
    return fig, ax


def plot_segmentation_prediction(image, box_pred, box_true=False):
    image *= 255
    image = image.astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if box_true:
        box_true *= 255
        box_true = box_true.astype('uint8')
        box_true = box_true.reshape(image.shape[0], image.shape[1], 1)
        cnts = cv2.findContours(box_true, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        true_cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        image_true = image.copy()
        cv2.drawContours(image_true, true_cnts, 0, (0, 191, 255), 2)

    box_pred = cv2.resize(box_pred, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_AREA)
    box_pred *= 255
    box_pred = box_pred.astype('uint8')
    box_pred = box_pred.reshape(image.shape[0], image.shape[1], 1)
    thresh = cv2.threshold(box_pred, 100, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    image_pred = image.copy()
    for j in range(len(pred_cnts)):
        cv2.drawContours(image_pred, pred_cnts, j, (0, 191, 255), 2)

    if box_true:
        vis = np.concatenate((image_true, image_pred), axis=1)
    else:
        vis = image_pred

    if box_true:
        heatmap_vis = np.concatenate((box_true, box_pred), axis=1)
    else:
        heatmap_vis = box_pred

    return vis, heatmap_vis


def pass_thresh(pred, threshold_matrix):
    flag = True
    pred_classes = np.argmax(pred, axis=1)
    for i in range(num_digits):
        if pred[i, pred_classes[i]] < threshold_matrix[i, pred_classes[i]]:
            flag = False
    return flag
