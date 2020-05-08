import cv2
import pickle
import h5py
import numpy as np

from src.config import hdf5_file_path
from src.config import num_digits, dataset_names, a_files_directory
from src.config import image_rows, image_cols
# from src.config import aug_image_rows, aug_image_cols
from src.config import splitted_dataset_list_path
from src.utils import digit_vector, char_vector
from src.utils import AFile
from src.DataPreparation.utils import label_to_output, read_plate_location, augment_image, map_smb_path_to_local


def create_hdf5_file(location_included):
    # Data groups
    groups = dataset_names
    with open(splitted_dataset_list_path, 'rb') as input:
        a_files = pickle.load(input)

    # Augment data or not
    # if augmentation:
    #     img_rows, img_cols = aug_image_rows, aug_image_cols
    #     hdf5_path = augmented_hdf5_path
    # else:
    img_rows, img_cols = image_rows, image_cols
    hdf5_path = hdf5_file_path

    # Define dataset shapes for each group
    data_shapes = {}
    for group in groups:
        data_shapes[group + '_image_shape'] = (len(a_files[group]), img_rows, img_cols, 1)
        data_shapes[group + '_label_shape'] = (
            len(a_files[group]),
            (num_digits - 1) * len(digit_vector) + len(char_vector))
        data_shapes[group + '_location_shape'] = (len(a_files[group]), img_rows, img_cols)
        data_shapes[group + '_afile_path_shape'] = (len(a_files[group]), 1)

    # open a hdf5 file and create datasets for each group
    hdf5_file = h5py.File(hdf5_path, mode='w')
    for group in groups:
        grp = hdf5_file.create_group(group)
        grp.create_dataset('images', data_shapes[group + '_image_shape'], np.float16)
        grp.create_dataset('labels', data_shapes[group + '_label_shape'], np.float16)
        if location_included:
            grp.create_dataset('locations', data_shapes[group + '_location_shape'], np.bool_)
        grp.create_dataset('a_file_paths', data_shapes[group + '_afile_path_shape'], 'S100')

    # loop over image addresses in each group and import to hdf5 file
    for group in groups:
        num_aug_images = 0
        for i in range(len(a_files[group])):
            if i % 1000 == 0 and i > 0:
                print('{} data: {}/{}'.format(group, i, len(a_files[group])))
            a_file = AFile(a_files_directory, a_files[group][i])
            image_path = (map_smb_path_to_local(a_file.address_lines['1338/bmp'][0].file))
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            label = np.array(label_to_output(a_file.get_label()))

            if location_included:
                location_path = image_path[:-3] + 'txt'
                box = read_plate_location(location_path)

                # if augmentation:
                #     img, box, stat = augment_image(img, box)
                #     if stat:
                #         num_aug_images += 1

                cimg = np.zeros_like(img)
                cv2.drawContours(cimg, [box], 0, color=255, thickness=-1)

                img = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
                img = img.astype('float32')
                img = img / 255
                img = img.reshape(img_rows, img_cols, 1)

                cimg = cimg / 255
                cimg = cv2.resize(cimg, (img_cols, img_rows), interpolation=cv2.INTER_NEAREST)
                cimg = cimg.astype('int')
            else:
                img = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
                img = img.astype('float32')
                img = img / 255
                img = img.reshape(img_rows, img_cols, 1)

            hdf5_file[group + "/images"][i, ...] = img[None]
            hdf5_file[group + "/labels"][i, ...] = label[None]
            if location_included:
                hdf5_file[group + "/locations"][i, ...] = cimg[None]
            hdf5_file[group + "/a_file_paths"][i, ...] = a_files[group][i].encode("ascii", "ignore")
            # if augmentation:
            #     print('Augmentation of {} images out of {} in group {} was successful'.format(num_aug_images,
            #                                                                                   len(a_files[group]),
            #                                                                                   group))
    hdf5_file.close()
