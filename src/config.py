images_path = '/home/train/ssd/Alireza/ALPR2/Data/Raw Data'
a_files_directory = '/home/train/ssd/Alireza/AFiles'
splitted_dataset_list_path = '/home/train/ssd/Alireza/ALPR2/Data/splitted_dataset.pkl'
augmented_hdf5_path = '/home/train/ssd/Alireza/ALPR2/Data/Set1-Set2-augmented.hdf5'
hdf5_file_path = '/home/train/ssd/Alireza/ALPR2/Data/Set1-Set2.hdf5'
local_raw_data_address = '/home/train/ssd/Alireza/ALPR2/Data/Raw Data/'
models_dirpath = 'Models/CrossValidation'
max_sample_size_for_each_char = 100000
dataset_names = ['Set1', 'Set2']
dataset_proportions = [0.5, 0.5]
image_rows, image_cols = 120, 360
aug_image_rows, aug_image_cols = 96, 288

smb_address = '\\\\192.168.0.1\\'
smb_mounted_address = '/home/train/smb/'

num_digits = 8
batch_size = 32
epochs = 10
drop_out = 0.25
workers = 20
multiprocessing = False
max_queue_size = 20
loss_weights = [0.8, 0.2]