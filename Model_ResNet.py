import numpy as np
from Evaluation import evaluation

"""
ResNet-18
Reference:
    [1] K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016
    [2] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
    Surpassing human-level performance on imagenet classification. In
    ICCV, 2015.
"""

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Layer
from keras.models import Sequential
from keras.models import Model
import tensorflow as tf


class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False, residual_path=False):
        """
        channels: same as the number of convolution kernels
        """
        super().__init__()

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in kKras), as suggested by He [2].
        INIT_SCHEME = "he_normal"
        strides = [1, 1] if down_sample else [2, 1]

        self.__channels = channels
        self.__strides = strides
        self.__residual_path = residual_path

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()

        if self.__residual_path:
            # perform down sampling using stride of 2, according to [1].
            self.down_conv = Conv2D(
                self.__channels, strides=self.__strides[0], kernel_size=(1, 1), kernel_initializer=INIT_SCHEME)
            self.down_bn = BatchNormalization()

    def call(self, inputs, training=None):
        residual = inputs

        x = self.bn_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)

        if self.__residual_path:
            residual = self.down_bn(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)

        x = x + residual
        return x


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64, residual_path=True)
        self.res_1_2 = ResnetBlock(64, residual_path=True)
        self.res_2_1 = ResnetBlock(128, residual_path=True, down_sample=True)
        self.res_2_2 = ResnetBlock(128, residual_path=True)
        self.res_3_1 = ResnetBlock(256, residual_path=True, down_sample=True)
        self.res_3_2 = ResnetBlock(256, residual_path=True)
        self.res_4_1 = ResnetBlock(512, residual_path=True, down_sample=True)
        self.res_4_2 = ResnetBlock(512, residual_path=True)
        self.final_bn = BatchNormalization()
        self.avg_pool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs, training=None):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2,
                          self.res_4_1, self.res_4_2]:
            out = res_block(out, training=training)
        out = self.final_bn(out, training=training)
        out = tf.nn.relu(out)
        out = self.avg_pool(out)
        out = self.fc(out)
        return out


def Model_ResNet18(train_data, train_target, test_data, test_target):
    IMG_SIZE = 32
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    model = ResNet18(train_target.shape[1])
    model.build(input_shape=(None, Train_X.shape[1], Train_X.shape[2], Train_X.shape[3]))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()
    model.fit(Train_X, train_target, epochs=100, batch_size=64, steps_per_epoch=200)
    pred = model.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(test_target, pred)

    return Eval, pred
