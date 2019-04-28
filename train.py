# -*- encoding:utf-8 -*-

import numpy as np
import tensorflow as tf
import time
import os

from utils import load_pickle
from vgg19 import VGG19

learning_rate = 1e-4  # 学习率
batch_size = 64  # batch_size
iter_num = 60  # 迭代次数
pre_vgg19_model = r"imagenet-vgg-verydeep-19.mat"  # 预训练模型
image_pkl = r"image.pkl"  # 图像矩阵
label_pkl = r"label.pkl"  # 标签矩阵


def train():
    imgs = load_pickle(image_pkl)
    labels = load_pickle(label_pkl)
    id_list = range(len(imgs))
    np.random.shuffle(id_list)

    mean = np.mean(imgs, axis=0)
    imgs = imgs - mean
    train_imgs = imgs[id_list[:int(0.7 * len(id_list))]]
    train_labels = labels[id_list[:int(0.7 * len(id_list))]]
    val_imgs = imgs[id_list[int(0.7 * len(id_list)):]]
    val_labels = labels[id_list[int(0.7 * len(id_list)):]]
    train_imgs = train_imgs
    train_labels = train_labels
    val_imgs = val_imgs
    val_labels = val_labels
    # train_labels = load_pickle(r"train.labels.pkl")[:50]
    print(train_imgs.shape)
    print(train_labels.shape)
    # val_imgs = load_pickle(r"train.imgs.pkl")[:20]

    # val_labels = load_pickle(r"train.labels.pkl")[:20]
    # print(val_imgs)
    # print(val_labels)
    model = VGG19(pre_vgg19_model)
    n_iters_per_epoch = int(np.ceil(float(train_imgs.shape[0]) / batch_size))
    n_iters_val = val_imgs.shape[0]
    with tf.variable_scope(tf.get_variable_scope()):
        loss = model.build_model()
        tf.get_variable_scope().reuse_variables()
        outputs = model.build_sample()
        correct_predict = tf.equal(tf.argmax(model.labels, dimension=1), outputs)
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads = tf.gradients(loss, tf.trainable_variables())
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        train_op = optimzer.apply_gradients(grads_and_vars=grads_and_vars)

    tf.summary.scalar("batch_loss", loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    for grad, var in grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradient", grad)

    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(r"logs",
                                               graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=10)
        prev_loss = -1
        curr_loss = 0
        start_t = time.time()

        for it in range(iter_num):
            rand_idxs = np.random.permutation(train_imgs.shape[0])
            train_labels = train_labels[rand_idxs]
            train_imgs = train_imgs[rand_idxs]
            train_total_acc = 0.0
            for i in range(n_iters_per_epoch):
                imgs_batch = train_imgs[i * batch_size: (i + 1) * batch_size]
                labels_batch = train_labels[i * batch_size: (i + 1) * batch_size]
                feed_dict = {model.images: imgs_batch, model.labels: labels_batch, model.keep_prob: 0.85}
                _, l, train_acc = sess.run([train_op, loss, accuracy], feed_dict)
                curr_loss = curr_loss + l
                train_total_acc = train_total_acc + train_acc
                if i % 10 == 0:
                    summary = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary, it * n_iters_per_epoch + i)
            print("Previous epoch loss: ", prev_loss / n_iters_per_epoch)
            print("Current epoch loss: ", curr_loss / n_iters_per_epoch)
            print("train epoch accuracy: ", train_total_acc / n_iters_per_epoch)
            print("Elapsed time: ", time.time() - start_t)
            prev_loss = curr_loss
            curr_loss = 0

            total_acc = 0.0
            for i in range(n_iters_val):
                imgs_batch = val_imgs[i:i + 1]
                labels_batch = val_labels[i:(i + 1)]
                feed_dict = {model.images: imgs_batch, model.labels: labels_batch, model.keep_prob: 1.0}
                acc = sess.run(accuracy, feed_dict)
                total_acc = total_acc + acc
            print("val epoch acc: ", total_acc / n_iters_val)

            if (it + 1) % 1 == 0:
                saver.save(sess,
                           os.path.join(r'model', 'model'),
                           global_step=it + 1)
                print("model-%s saved." % (it + 1))


if __name__ == "__main__":
    train()
