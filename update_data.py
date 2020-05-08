from src.DataPreparation.create_hdf5_file import create_hdf5_file
from src.DataPreparation.update_splitted_data import update_splitted_data

# Location include in data or data just prepared for OCR purpose
location_included = False

if __name__ == '__main__':
    # Update Splitted data list (eliminate noises)
    update_splitted_data()
    # Create HDF5 file
    create_hdf5_file(location_included)

