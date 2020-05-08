import h5py
import numpy as np

from src.config import hdf5_file_path
from src.config import dataset_names, a_files_directory
from src.utils import AFile
from src.DataPreparation.utils import label_to_output
from src.config import augmented_hdf5_path, hdf5_file_path

def update_hdf5_file(augmentation):
    # Data groups
    groups = dataset_names

    # Update Augmented data or non-augmented data
    if augmentation:
        hdf5_path = augmented_hdf5_path
    else:
        hdf5_path = hdf5_path

    # open a hdf5 file and create datasets for each group
    hdf5_file = h5py.File(hdf5_path, mode='r+')

    # loop over image addresses in each group and import to hdf5 file
    for group in groups:
        for i in range(hdf5_file[group]['a_file_paths'].shape[0]):
            if i % 1000 == 0 and i > 0:
                print('{} data: {}/{}'.format(group, i, hdf5_file[group]['a_file_paths'].shape[0]))
            a_file_name = str(hdf5_file[group + '/a_file_paths'][i][0])[2:-3] + '.a'
            a_file = AFile(a_files_directory, a_file_name)

            label = np.array(label_to_output(a_file.get_label()))

            hdf5_file[group + "/labels"][i, ...] = label[None]
    hdf5_file.close()
