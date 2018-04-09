import os
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors

from settings import FILES_DIR, VGG_19_CHECKPOINT_FILENAME, VGG_19_CODE_LAYER, IMAGE_DATASET_PATH, IMAGE_SIZE
from vgg import vgg_19
from prepare import rescale_image


def get_images_codes(images, images_placeholder, end_points):
    batch_size = 4
    saver = tf.train.Saver(tf.get_collection('model_variables'))
    with tf.Session() as sess:
        saver.restore(sess, VGG_19_CHECKPOINT_FILENAME)
        codes = None
        for i in range(0, images.shape[0], batch_size):
            batch_images = images[i:i + batch_size, ...]
            batch_codes = sess.run(end_points[VGG_19_CODE_LAYER], feed_dict={images_placeholder: batch_images})
            if codes is None:
                codes = batch_codes
            else:
                codes = np.concatenate((codes, batch_codes))

        return np.squeeze(codes, axis=(1, 2))


def get_dataset_image_codes(images_placeholder, end_points):
    files = [os.path.join(IMAGE_DATASET_PATH, f) for f in os.listdir(IMAGE_DATASET_PATH)]
    images = np.stack([np.asarray(Image.open(f)) for f in files])

    image_codes = get_images_codes(images, images_placeholder, end_points)
    return image_codes, files


def get_query_image_code(filenames, images_placeholder, end_points):
    images = np.stack([np.asarray(rescale_image(Image.open(f))) for f in filenames])
    image_codes = get_images_codes(images, images_placeholder, end_points)
    return image_codes


def main():
    images_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    _, end_points = vgg_19(images_placeholder, num_classes=None, is_training=False)

    dataset_image_codes, dataset_image_files = get_dataset_image_codes(images_placeholder, end_points)
    print(dataset_image_codes.shape)

    images = [os.path.join(FILES_DIR, f'image_{i}.jpg') for i in range(1, 5)]
    query_image_codes = get_query_image_code(images, images_placeholder, end_points)
    print(query_image_codes.shape)

    neighbors_count = 2
    nearest_neighbors = NearestNeighbors(n_neighbors=neighbors_count, metric='cosine').fit(dataset_image_codes)
    _, indices = nearest_neighbors.kneighbors(query_image_codes)

    space = 10
    result_image_size = (
        (neighbors_count + 1) * (IMAGE_SIZE + space) - space,
        len(images) * (IMAGE_SIZE + space) - space
    )

    result_image = Image.new('RGB', result_image_size, 'white')
    for i, filename in enumerate(images):
        query_image = rescale_image(Image.open(filename))
        draw = ImageDraw.Draw(query_image)
        draw.line(
            (
                0, 0,
                query_image.width - 1, 0,
                query_image.width - 1, query_image.height - 1,
                0, query_image.height - 1,
                0, 0
            ),
            fill='red', width=1)
        result_image.paste(query_image, (0, i * (IMAGE_SIZE + space)))
        for j in range(neighbors_count):
            neighbor_image = Image.open(dataset_image_files[indices[i][j]])
            result_image.paste(neighbor_image, ((j + 1) * (IMAGE_SIZE + space), i * (IMAGE_SIZE + space)))

    result_image.show()
    result_image.save(os.path.join(FILES_DIR, 'result.jpg'))


if __name__ == '__main__':
    main()
