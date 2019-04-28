# -*- encoding:utf-8 -*-
import tensorflow as tf
import scipy.io



"""
vggnet 进行遥感影像场景分类
"""


vgg_layers = [
    "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
    "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
    "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "pool3",
    "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "pool4",
    "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "pool5"
    "fc6",
    "fc7"
]


class VGG19(object):
    def __init__(self, vgg_path=None):
        self.vgg_path = vgg_path
        self.build_inputs()
        self.build_params()

    def build_inputs(self):
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], "images")
        self.labels = tf.placeholder(tf.float32, [None, 12], "labels")
        self.keep_prob = tf.placeholder(tf.float32)

    def build_params(self):
        if self.vgg_path:
            model = scipy.io.loadmat(self.vgg_path)
            layers = model["layers"][0]
            self.params = {}
            for i, layer in enumerate(layers):
                layer_name = layer[0][0][0][0]
                layer_type = layer[0][0][1][0]
                if layer_type == "conv":
                    w = layer[0][0][2][0][0].transpose(1, 0, 2, 3)
                    b = layer[0][0][2][0][1].reshape(-1)

                    if layer_name == "fc7":
                        self.params["fc7"] = {}
                        self.params["fc7"]["w"] = self.get_weight(name="fc7_w", shape=[4096, 12])
                        self.params["fc7"]["b"] = self.get_biases(name="fc7_b", length=12)
                        break
                    elif layer_name == "fc6":
                        self.params["fc6"] = {}
                        self.params["fc6"]["w"] = self.get_weight(name="fc6_w", shape=[7 * 7 * 512, 4096])
                        self.params["fc6"]["b"] = self.get_biases(name="fc6_b", length=4096)

                    else:
                        self.params[layer_name] = {}
                        self.params[layer_name]["w"] = tf.get_variable(name=layer_name + "_w",
                                                                       initializer=tf.constant(w),
                                                                       trainable=True)
                        self.params[layer_name]["b"] = tf.get_variable(name=layer_name + "_b",
                                                                       initializer=tf.constant(b),
                                                                       trainable=True)
                    print(layer_name, w.shape, b.shape)
        else:
            self.params = {}
            self.params["conv1_1"] = {}
            self.params["conv1_1"]["w"] = self.get_weight(name="conv1_1_w", shape=[3, 3, 3, 64])
            self.params["conv1_1"]["b"] = self.get_biases(name="conv1_1_b", length=64)
            self.params["conv1_2"] = {}
            self.params["conv1_2"]["w"] = self.get_weight(name="conv1_2_w", shape=[3, 3, 64, 64])
            self.params["conv1_2"]["b"] = self.get_biases(name="conv1_2_b", length=64)

            self.params["conv2_1"] = {}
            self.params["conv2_1"]["w"] = self.get_weight(name="conv2_1_w", shape=[3, 3, 64, 128])
            self.params["conv2_1"]["b"] = self.get_biases(name="conv2_1_b", length=128)
            self.params["conv2_2"] = {}
            self.params["conv2_2"]["w"] = self.get_weight(name="conv2_2_w", shape=[3, 3, 128, 128])
            self.params["conv2_2"]["b"] = self.get_biases(name="conv2_2_b", length=128)

            self.params["conv3_1"] = {}
            self.params["conv3_1"]["w"] = self.get_weight(name="conv3_1_w", shape=[3, 3, 128, 256])
            self.params["conv3_1"]["b"] = self.get_biases(name="conv3_1_b", length=256)
            self.params["conv3_2"] = {}
            self.params["conv3_2"]["w"] = self.get_weight(name="conv3_2_w", shape=[3, 3, 256, 256])
            self.params["conv3_2"]["b"] = self.get_biases(name="conv3_2_b", length=256)
            self.params["conv3_3"] = {}
            self.params["conv3_3"]["w"] = self.get_weight(name="conv3_3_w", shape=[3, 3, 256, 256])
            self.params["conv3_3"]["b"] = self.get_biases(name="conv3_3_b", length=256)
            self.params["conv3_4"] = {}
            self.params["conv3_4"]["w"] = self.get_weight(name="conv3_4_w", shape=[3, 3, 256, 256])
            self.params["conv3_4"]["b"] = self.get_biases(name="conv3_4_b", length=256)

            self.params["conv4_1"] = {}
            self.params["conv4_1"]["w"] = self.get_weight(name="conv4_1_w", shape=[3, 3, 256, 512])
            self.params["conv4_1"]["b"] = self.get_biases(name="conv4_1_b", length=512)
            self.params["conv4_2"] = {}
            self.params["conv4_2"]["w"] = self.get_weight(name="conv4_2_w", shape=[3, 3, 512, 512])
            self.params["conv4_2"]["b"] = self.get_biases(name="conv4_2_b", length=512)
            self.params["conv4_3"] = {}
            self.params["conv4_3"]["w"] = self.get_weight(name="conv4_3_w", shape=[3, 3, 512, 512])
            self.params["conv4_3"]["b"] = self.get_biases(name="conv4_3_b", length=512)
            self.params["conv4_4"] = {}
            self.params["conv4_4"]["w"] = self.get_weight(name="conv4_4_w", shape=[3, 3, 512, 512])
            self.params["conv4_4"]["b"] = self.get_biases(name="conv4_4_b", length=512)

            self.params["conv5_1"] = {}
            self.params["conv5_1"]["w"] = self.get_weight(name="conv5_1_w", shape=[3, 3, 512, 512])
            self.params["conv5_1"]["b"] = self.get_biases(name="conv5_1_b", length=512)
            self.params["conv5_2"] = {}
            self.params["conv5_2"]["w"] = self.get_weight(name="conv5_2_w", shape=[3, 3, 512, 512])
            self.params["conv5_2"]["b"] = self.get_biases(name="conv5_2_b", length=512)
            self.params["conv5_3"] = {}
            self.params["conv5_3"]["w"] = self.get_weight(name="conv5_3_w", shape=[3, 3, 512, 512])
            self.params["conv5_3"]["b"] = self.get_biases(name="conv5_3_b", length=512)
            self.params["conv5_4"] = {}
            self.params["conv5_4"]["w"] = self.get_weight(name="conv5_4_w", shape=[3, 3, 512, 512])
            self.params["conv5_4"]["b"] = self.get_biases(name="conv5_4_b", length=512)
            self.params["fc6"] = {}
            self.params["fc6"]["w"] = self.get_weight(name="fc6_w", shape=[7 * 7 * 512, 12])
            self.params["fc6"]["b"] = self.get_biases(name="fc6_b", length=12)

    def get_weight(self, shape, name):
        return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.01), name=name)

    def get_biases(self, length, name):
        return tf.get_variable(shape=[length], initializer=tf.constant_initializer(0.05), name=name)

    def _conv(self, x, w, b):
        return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME"), b)

    def _relu(self, x):
        return tf.nn.relu(x)

    def _pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    def build_model(self):
        self.conv1_1 = self._conv(self.images, self.params["conv1_1"]["w"], self.params["conv1_1"]["b"])
        self.relu1_1 = self._relu(self.conv1_1)
        self.conv1_2 = self._conv(self.relu1_1, self.params["conv1_2"]["w"], self.params["conv1_2"]["b"])
        self.relu1_2 = self._relu(self.conv1_2)
        self.pool1 = self._pool(self.relu1_2)

        self.conv2_1 = self._conv(self.pool1, self.params["conv2_1"]["w"], self.params["conv2_1"]["b"])
        self.relu2_1 = self._relu(self.conv2_1)
        self.conv2_2 = self._conv(self.relu2_1, self.params["conv2_2"]["w"], self.params["conv2_2"]["b"])
        self.relu2_2 = self._relu(self.conv2_2)
        self.pool2 = self._pool(self.relu2_2)

        self.conv3_1 = self._conv(self.pool2, self.params["conv3_1"]["w"], self.params["conv3_1"]["b"])
        self.relu3_1 = self._relu(self.conv3_1)
        self.conv3_2 = self._conv(self.relu3_1, self.params["conv3_2"]["w"], self.params["conv3_2"]["b"])
        self.relu3_2 = self._relu(self.conv3_2)
        self.conv3_3 = self._conv(self.relu3_2, self.params["conv3_3"]["w"], self.params["conv3_3"]["b"])
        self.relu3_3 = self._relu(self.conv3_3)
        self.conv3_4 = self._conv(self.relu3_3, self.params["conv3_4"]["w"], self.params["conv3_4"]["b"])
        self.relu3_4 = self._relu(self.conv3_4)
        self.pool3 = self._pool(self.relu3_4)

        self.conv4_1 = self._conv(self.pool3, self.params["conv4_1"]["w"], self.params["conv4_1"]["b"])
        self.relu4_1 = self._relu(self.conv4_1)
        self.conv4_2 = self._conv(self.relu4_1, self.params["conv4_2"]["w"], self.params["conv4_2"]["b"])
        self.relu4_2 = self._relu(self.conv4_2)
        self.conv4_3 = self._conv(self.relu4_2, self.params["conv4_3"]["w"], self.params["conv4_3"]["b"])
        self.relu4_3 = self._relu(self.conv4_3)
        self.conv4_4 = self._conv(self.relu4_3, self.params["conv4_4"]["w"], self.params["conv4_4"]["b"])
        self.relu4_4 = self._relu(self.conv4_4)
        self.pool4 = self._pool(self.relu4_4)

        self.conv5_1 = self._conv(self.pool4, self.params["conv5_1"]["w"], self.params["conv5_1"]["b"])
        self.relu5_1 = self._relu(self.conv5_1)
        self.conv5_2 = self._conv(self.relu5_1, self.params["conv5_2"]["w"], self.params["conv5_2"]["b"])
        self.relu5_2 = self._relu(self.conv5_2)
        self.conv5_3 = self._conv(self.relu5_2, self.params["conv5_3"]["w"], self.params["conv5_3"]["b"])
        self.relu5_3 = self._relu(self.conv5_3)
        self.conv5_4 = self._conv(self.relu5_3, self.params["conv5_4"]["w"], self.params["conv5_4"]["b"])
        self.relu5_4 = self._relu(self.conv5_4)
        self.pool5 = self._pool(self.relu5_4)
        print(self.pool5.shape)
        self.features = tf.reshape(self.pool5, [-1, 49 * 512])
        self.fc6 = tf.matmul(self.features, self.params["fc6"]["w"]) + self.params["fc6"]["b"]
        self.relu6 = self._relu(self.fc6)
        self.relu6 = tf.nn.dropout(self.relu6, self.keep_prob)
        self.fc7 = tf.matmul(self.relu6, self.params["fc7"]["w"]) + self.params["fc7"]["b"]
        self.fc7 = tf.nn.dropout(self.fc7, self.keep_prob)
        print(self.fc7.shape)
        # self.output = tf.matmul(self.fc6, self.get_weight([4096, 7], "output_w")) + self.get_biases(7, "output_b")
        # print(self.output.shape)
        # # self.output = tf.nn.softmax(self.output)
        coss_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc7, labels=self.labels)
        # coss_entropy = - tf.reduce_sum(self.labels * tf.log(tf.clip_by_value(self.fc6,1e-10,1.0)))
        loss = tf.reduce_mean(coss_entropy)
        return loss

    def build_sample(self):
        self.conv1_1 = self._conv(self.images, self.params["conv1_1"]["w"], self.params["conv1_1"]["b"])
        self.relu1_1 = self._relu(self.conv1_1)
        self.conv1_2 = self._conv(self.relu1_1, self.params["conv1_2"]["w"], self.params["conv1_2"]["b"])
        self.relu1_2 = self._relu(self.conv1_2)
        self.pool1 = self._pool(self.relu1_2)

        self.conv2_1 = self._conv(self.pool1, self.params["conv2_1"]["w"], self.params["conv2_1"]["b"])
        self.relu2_1 = self._relu(self.conv2_1)
        self.conv2_2 = self._conv(self.relu2_1, self.params["conv2_2"]["w"], self.params["conv2_2"]["b"])
        self.relu2_2 = self._relu(self.conv2_2)
        self.pool2 = self._pool(self.relu2_2)

        self.conv3_1 = self._conv(self.pool2, self.params["conv3_1"]["w"], self.params["conv3_1"]["b"])
        self.relu3_1 = self._relu(self.conv3_1)
        self.conv3_2 = self._conv(self.relu3_1, self.params["conv3_2"]["w"], self.params["conv3_2"]["b"])
        self.relu3_2 = self._relu(self.conv3_2)
        self.conv3_3 = self._conv(self.relu3_2, self.params["conv3_3"]["w"], self.params["conv3_3"]["b"])
        self.relu3_3 = self._relu(self.conv3_3)
        self.conv3_4 = self._conv(self.relu3_3, self.params["conv3_4"]["w"], self.params["conv3_4"]["b"])
        self.relu3_4 = self._relu(self.conv3_4)
        self.pool3 = self._pool(self.relu3_4)

        self.conv4_1 = self._conv(self.pool3, self.params["conv4_1"]["w"], self.params["conv4_1"]["b"])
        self.relu4_1 = self._relu(self.conv4_1)
        self.conv4_2 = self._conv(self.relu4_1, self.params["conv4_2"]["w"], self.params["conv4_2"]["b"])
        self.relu4_2 = self._relu(self.conv4_2)
        self.conv4_3 = self._conv(self.relu4_2, self.params["conv4_3"]["w"], self.params["conv4_3"]["b"])
        self.relu4_3 = self._relu(self.conv4_3)
        self.conv4_4 = self._conv(self.relu4_3, self.params["conv4_4"]["w"], self.params["conv4_4"]["b"])
        self.relu4_4 = self._relu(self.conv4_4)
        self.pool4 = self._pool(self.relu4_4)

        self.conv5_1 = self._conv(self.pool4, self.params["conv5_1"]["w"], self.params["conv5_1"]["b"])
        self.relu5_1 = self._relu(self.conv5_1)
        self.conv5_2 = self._conv(self.relu5_1, self.params["conv5_2"]["w"], self.params["conv5_2"]["b"])
        self.relu5_2 = self._relu(self.conv5_2)
        self.conv5_3 = self._conv(self.relu5_2, self.params["conv5_3"]["w"], self.params["conv5_3"]["b"])
        self.relu5_3 = self._relu(self.conv5_3)
        self.conv5_4 = self._conv(self.relu5_3, self.params["conv5_4"]["w"], self.params["conv5_4"]["b"])
        self.relu5_4 = self._relu(self.conv5_4)
        self.pool5 = self._pool(self.relu5_4)
        self.features = tf.reshape(self.pool5, [-1, 49 * 512])

        self.fc6 = tf.matmul(self.features, self.params["fc6"]["w"]) + self.params["fc6"]["b"]
        self.relu6 = self._relu(self.fc6)
        self.relu6 = tf.nn.dropout(self.relu6, self.keep_prob)
        self.fc7 = tf.matmul(self.relu6, self.params["fc7"]["w"]) + self.params["fc7"]["b"]
        self.fc7 = tf.nn.dropout(self.fc7, self.keep_prob)
        print(self.fc7.shape)
        # self.output = tf.matmul(self.fc6, self.get_weight([4096, 7], "output_w")) + self.get_biases(7, "output_b")
        self.output = tf.nn.softmax(self.fc7)
        self.predict_cls = tf.argmax(self.output, dimension=1)
        return self.predict_cls

    def build(self):
        self.build_inputs()
        self.build_params()
        self.build_model()