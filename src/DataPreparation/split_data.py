import os
import pickle
import random

from src.utils import char_vector, AFile
from src.utils import digit_dictionary, digit_char_vector
from src.DataPreparation.utils import get_confidence_from_filename
from src.config import splitted_dataset_list_path, a_files_directory
from src.config import dataset_proportions, dataset_names, max_sample_size_for_each_char


def split_data():
    a_files_found = 0
    a_files_added = 0
    # Get Images Paths
    a_files = {}
    for char in char_vector:
        a_files[char] = []
    for (dirpath, dirnames, filenames) in os.walk(a_files_directory):
        for file in filenames:
            if os.path.isfile(os.path.join(dirpath, file)) and \
                            file.split('.')[1] == 'a' and \
                            get_confidence_from_filename(file) > 20:
                if a_files_found % 10000 == 0 and a_files_found > 0:
                    print('{} a files processed / {} added to data'.format(a_files_found, a_files_added))
                a_files_found += 1
                a_file = AFile(dirpath, file)
                platetype = a_file.lines['PlateType'].tag
                char = a_file.lines['3'].tag
                valid_digits = True
                for index in ['1', '2', '4', '5', '6', '7', '8']:
                    if digit_dictionary[a_file.lines[index].tag] not in digit_char_vector:
                        valid_digits = False
                if platetype == 'ok' and valid_digits and char in char_vector:
                    a_files[char].append(file)
                    a_files_added += 1

    print('------------------------------------')
    print('{} a files processed / {} added to data'.format(a_files_found, a_files_added))
    print('------------------------------------')

    for key in a_files:
        print('{} : {} valid afiles found'.format(key, len(a_files[key])))

    selected_a_files = []
    for key in a_files.keys():
        random.shuffle(a_files[key])
        if len(a_files[key]) <= max_sample_size_for_each_char:
            selected_a_files += a_files[key]
        else:
            selected_a_files += random.sample(a_files[key], max_sample_size_for_each_char)

    # Split train-val-test
    random.shuffle(selected_a_files)
    cuts = []
    for i in range(len(dataset_proportions)):
        if len(cuts) == 0:
            cuts.append(int(dataset_proportions * len(selected_a_files)))
        elif i == len(dataset_proportions):
            cuts.append(len(selected_a_files))
        else:
            cuts.append(int(dataset_proportions * len(selected_a_files)) + cuts[-1])

    dataset = {}
    for i in range(dataset_names):
        name = dataset_names[i]
        if i == 0 :
            cut = [0, cuts[0]]
        else:
            cut = [cuts[i-1], cuts[i]]
        dataset[name] = selected_a_files[cut[0]:cut[1]]

    with open(splitted_dataset_list_path, 'wb') as output:
        pickle.dump(dataset, output)