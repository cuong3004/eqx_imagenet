import tensorflow as tf
from tensorflow import keras

class Augmentation(keras.layers.Layer):


    def __init__(self):
        super().__init__()

    @tf.function
    def random_execute(self, prob: float) -> bool:

        return tf.random.uniform([], minval=0, maxval=1) < prob


class RandomToGrayscale(Augmentation):

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:

        if self.random_execute(0.2):
            x = tf.image.rgb_to_grayscale(x)
            x = tf.tile(x, [1, 1, 3])
        return x


class RandomColorJitter(Augmentation):

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:

        if self.random_execute(0.8):
            x = tf.image.random_brightness(x, 0.8)
            x = tf.image.random_contrast(x, 0.4, 1.6)
            x = tf.image.random_saturation(x, 0.4, 1.6)
            x = tf.image.random_hue(x, 0.2)
        return x


class RandomFlip(Augmentation):

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:

        if self.random_execute(0.5):
            x = tf.image.random_flip_left_right(x)
        return x


class RandomResizedCrop(Augmentation):

    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def call(self, x: tf.Tensor) -> tf.Tensor:


        rand_size = tf.random.uniform(
            shape=[],
            minval=int(0.75 * self.image_size),
            maxval=1 * self.image_size,
            dtype=tf.int32,
        )

        crop = tf.image.random_crop(x, (rand_size, rand_size, 3))
        crop_resize = tf.image.resize(crop, (self.image_size, self.image_size))
        return crop_resize


class RandomSolarize(Augmentation):

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:


        if self.random_execute(0.2):
            # flips abnormally low pixels to abnormally high pixels
            x = tf.where(x < 10, x, 255 - x)
        return x


class RandomBlur(Augmentation):

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:

        if self.random_execute(0.2):
            s = np.random.random()
            return tfa.image.gaussian_filter2d(image=x, sigma=s)
        return x


class TrainRandomAugmentor(keras.Model):

    def __init__(self, image_size: int):
        super().__init__()

        self.image_size = image_size
        self.random_resized_crop = RandomResizedCrop(image_size)
        self.random_flip = RandomFlip()
        self.random_color_jitter = RandomColorJitter()
        self.random_blur = RandomBlur()
        self.random_to_grayscale = RandomToGrayscale()
        self.random_solarize = RandomSolarize()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.image.resize(x, (int(self.image_size*1.2), int(self.image_size*1.2)))
        x = self.random_resized_crop(x)
        x = self.random_flip(x)
        x = self.random_color_jitter(x)
        # x = self.random_blur(x)
        # x = self.random_to_grayscale(x)
        # x = self.random_solarize(x)
        x = x/255.0
        # x = tf.clip_by_value(x, 0, 1)
        return x
    
class ValidRandomAugmentor(keras.Model):

    def __init__(self, image_size: int):
        super().__init__()

        self.image_size = image_size
        self.random_resized_crop = RandomResizedCrop(image_size)
        self.random_flip = RandomFlip()
        self.random_color_jitter = RandomColorJitter()
        self.random_blur = RandomBlur()
        self.random_to_grayscale = RandomToGrayscale()
        self.random_solarize = RandomSolarize()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x = self.random_resized_crop(x)
        x = tf.image.resize(x, (self.image_size, self.image_size))
        x = x/255.0
        return x
    
# def preprocessing(sample):


