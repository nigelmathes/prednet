'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
from scipy.misc import imread, imresize
import hickle as hkl
from settings import *


desired_image_size = (128, 160)
#categories = ['in_plane', 'outside_plane']
categories = ['default']

# Videos used for validation and testing.
val_recordings = [('city', '2011_09_26_drive_0005_sync')]
test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    # Assign videos to training and testing. Cross-validation done across entire recordings.
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    for category in categories:
        category_dir = os.path.join(DATA_DIR, 'raw', category + '/')
        _, video_id_dirs, _ = os.walk(category_dir).next()
        splits['train'] += [(category, video_id_dir) for video_id_dir in video_id_dirs
                            if (category, video_id_dir) not in not_train]

    # Create lists of images and their source videos
    for split in splits:
        image_list = []
        source_list = []
        for category, video_id_dir in splits[split]:
            _, _, image_files = os.walk(video_id_dir).next()
            image_list += [video_id_dir + f for f in sorted(image_files)]
            source_list += [category + '-' + video_id_dir] * len(image_files)

        # Dump the images and their source information to hickle files
        image_output_holder = np.zeros((len(image_list),) + desired_image_size + (3,), np.uint8)
        for i, image_file in enumerate(image_list):
            image = imread(image_file)
            image_output_holder[i] = process_image(image, desired_image_size)

        hkl.dump(image_output_holder, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))


# resize and crop image
def process_image(image, desired_size):
    target_size = float(desired_size[0])/image.shape[0]
    image = imresize(image, (desired_size[0], int(np.round(target_size * image.shape[1]))))
    d = int((image.shape[1] - desired_size[1]) / 2)
    image = image[:, d:d+desired_size[1]]
    return image


if __name__ == '__main__':
    process_data()
