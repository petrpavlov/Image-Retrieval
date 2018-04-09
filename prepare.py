import argparse
import os
import logging
import requests
import tarfile

import numpy as np

from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, wait
from PIL import Image
from sklearn.utils import shuffle

from settings import FILES_DIR, VGG_19_CHECKPOINT_URL, VGG_19_CHECKPOINT_FILENAME, IMAGE_DATASET_PATH, \
    IMAGE_DATASET_SIZE, IMAGE_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare environment for other modules.')

    parser.add_argument('dataset', type=str, help='Path to dataset')

    return vars(parser.parse_args())


def prepare_vgg_19_checkpoint():
    if not os.path.exists(VGG_19_CHECKPOINT_FILENAME):
        logging.info(f'Checkpoint does not exist. Download from: {VGG_19_CHECKPOINT_URL}')
        response = requests.get(VGG_19_CHECKPOINT_URL)

        logging.info(f'Extract checkpoint into {FILES_DIR}')
        with tarfile.open(fileobj=BytesIO(response.content)) as tar:
            tar.extractall(FILES_DIR)
    else:
        logging.info('Checkpoint already exists')


def rescale_image(image):
    size = np.asarray(image.size)
    size = (size * IMAGE_SIZE / min(size)).astype(int)
    image = image.resize(size, resample=Image.LANCZOS)
    w, h = image.size
    image = image.crop((
        (w - IMAGE_SIZE) // 2,
        (h - IMAGE_SIZE) // 2,
        (w + IMAGE_SIZE) // 2,
        (h + IMAGE_SIZE) // 2)
    )
    return image


def prepare_dataset(source_path):
    logging.info(f'Make dataset from {source_path}')

    def prepare_image(filename):
        image = Image.open(os.path.join(source_path, filename))
        if image.mode == 'RGB':
            image = rescale_image(image)
            image.save(os.path.join(IMAGE_DATASET_PATH, filename))

    if not os.path.exists(IMAGE_DATASET_PATH):
        os.mkdir(IMAGE_DATASET_PATH)

        files = shuffle([f for f in os.listdir(source_path) if os.path.splitext(f)[1] == '.jpg'])
        files = files[:IMAGE_DATASET_SIZE]

        executor = ThreadPoolExecutor()
        batch_size = 2
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            futures = [executor.submit(prepare_image, filename) for filename in batch]
            wait(futures)

        logging.info('Dataset successfully prepared')
    else:
        logging.info('Dataset already exests')


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    args = parse_args()
    prepare_vgg_19_checkpoint()
    prepare_dataset(args['dataset'])


if __name__ == '__main__':
    main()
