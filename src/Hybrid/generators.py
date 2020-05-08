import numpy as np
import gc
from keras.utils import Sequence


class HybridGenerator(Sequence):
    def __init__(self, images, labels, locations, batch_size):
        self.images = images
        self.labels = labels
        self.locations = locations
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_ocr = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_seg = self.locations[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_seg = [label.flatten() for label in batch_y_seg]
        gc.collect()
        return np.array(batch_x), {'OCR': np.array(batch_y_ocr),
                                   'Segmentation': np.array(batch_y_seg)}


class OCRGenerator(Sequence):
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_ocr = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        gc.collect()
        return np.array(batch_x), np.array(batch_y_ocr)


class SegmentationGenerator(Sequence):
    def __init__(self, images, locations, batch_size):
        self.images = images
        self.locations = locations
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_seg = self.locations[idx * self.batch_size: (idx + 1) * self.batch_size]
        gc.collect()
        return np.array(batch_x), np.array(batch_y_seg)