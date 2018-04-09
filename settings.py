import os


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
if not os.path.exists(FILES_DIR):
    os.mkdir(FILES_DIR)

VGG_19_CHECKPOINT_URL = 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'
VGG_19_CHECKPOINT_FILENAME = os.path.join(FILES_DIR, 'vgg_19.ckpt')
VGG_19_CODE_LAYER = 'vgg_19/fc7'

IMAGE_DATASET_PATH = os.path.join(FILES_DIR, 'dataset')
IMAGE_DATASET_SIZE = 3000
IMAGE_SIZE = 224
