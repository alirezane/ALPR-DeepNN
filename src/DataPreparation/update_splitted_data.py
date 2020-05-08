import pickle
import h5py

from src.config import hdf5_file_path
from src.config import dataset_names, a_files_directory
from src.config import splitted_dataset_list_path
from src.utils import digit_dictionary, digit_char_vector, char_vector
from src.utils import AFile

def update_splitted_data():
    # Data groups
    groups = dataset_names

    # Augment data or not
    hdf5_path = hdf5_file_path

    # open a hdf5 file and create datasets for each group
    hdf5_file = h5py.File(hdf5_path, mode='r')

    dataset = {}
    for group in groups:
        dataset[group] = []
        for i in range(hdf5_file[group]['a_file_paths'].shape[0]):
            if i % 10000 == 0 and i > 0:
                print('{} data: {} / {} processed {} added'.format(group,
                                                                   i,
                                                                   hdf5_file[group]['a_file_paths'].shape[0],
                                                                   len(dataset[group])))
            a_file_name = str(hdf5_file[group + '/a_file_paths'][i][0])[2:-3] + '.a'
            a_file = AFile(a_files_directory, a_file_name)
            platetype = a_file.lines['PlateType'].tag
            char = a_file.lines['3'].tag
            valid_digits = True
            for index in ['1', '2', '4', '5', '6', '7', '8']:
                if digit_dictionary[a_file.lines[index].tag] not in digit_char_vector:
                    valid_digits = False
            if platetype == 'ok' and valid_digits and char in char_vector:
                dataset[group].append(a_file_name)

    with open(splitted_dataset_list_path, 'wb') as output:
        pickle.dump(dataset, output)