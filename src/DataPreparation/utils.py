import random
import cv2
import numpy as np

from src.config import image_rows,image_cols, aug_image_rows, aug_image_cols
from src.config import num_digits, smb_address, local_raw_data_address
from src.utils import digit_vector, char_vector


def augment_image(img, box):
    image = img.copy()
    bbox = box.copy()
    aug_frame_x = int((aug_image_cols / image_cols) * image.shape[1])
    aug_frame_y = int((aug_image_rows / image_rows) * image.shape[0])
    counter = 0
    while True:
        if counter < 100:
            random_center_x = random.randint(int(aug_frame_x / 2) + 2,
                                             image.shape[1] - (int(aug_frame_x / 2) + 2))
            random_center_y = random.randint(int(aug_frame_y / 2) + 2,
                                             image.shape[0] - (int(aug_frame_y / 2) + 2))
            x_max_bound = random_center_x + int(aug_frame_x / 2)
            x_min_bound = random_center_x - int(aug_frame_x / 2)
            y_max_bound = random_center_y + int(aug_frame_y / 2)
            y_min_bound = random_center_y - int(aug_frame_y / 2)
            if x_min_bound < bbox[:, 0].max() < x_max_bound \
                    and y_min_bound < bbox[:, 1].max() < y_max_bound:
                image = image[y_min_bound:y_max_bound, x_min_bound:x_max_bound]
                bbox[:, 0] = bbox[:, 0] - x_min_bound
                bbox[:, 1] = bbox[:, 1] - y_min_bound
                return image, bbox, True
            else:
                counter += 1
        else:
            print('could not augment image')
            return image, bbox, False


def label_to_output(label):
    l = list(map(lambda x: digit_vector.index(x), label[:2]))
    l += [char_vector.index(label[2])]
    l += list(map(lambda x: digit_vector.index(x), label[3:]))
    l_cat = []
    for i in range(num_digits):
        if i != 2:
            l_cat += np.eye(len(digit_vector))[l[i]].tolist()
        else:
            l_cat += np.eye(len(char_vector))[l[i]].tolist()
    l_cat = np.array(l_cat)
    return l_cat


def read_plate_location(label_path):
    f = open(label_path, 'r')
    line = f.readline()
    f.close()
    line = np.array(','.join(line.split(';')[:-1]).split(',')).astype(int)
    line = line.reshape(int(line.shape[0] / 2), 1, 2)
    rect = cv2.minAreaRect(line)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    return box


def get_confidence_from_filename(filename):
    return int(filename.split('-')[7])


def map_smb_path_to_local(path):
    new_path = path.replace(smb_address, local_raw_data_address)
    new_path = new_path.replace('\\', '/')
    return new_path
