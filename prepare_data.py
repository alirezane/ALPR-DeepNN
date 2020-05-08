from src.DataPreparation.split_data import split_data
from src.DataPreparation.create_hdf5_file import create_hdf5_file

# Include location in data or prepare data just for OCR
location_included = False

if __name__ == '__main__':
    # Split Train Test Validation files
    split_data()
    # Create HDF5 file
    create_hdf5_file(location_included)

