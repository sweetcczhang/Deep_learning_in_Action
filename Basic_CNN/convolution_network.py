#!/usr/local/miniconda2/bin/python
# _*_ coding:utf-8 _*_

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class ConvolutionNetwork(object):
    def __init__(self,image_height, image_width, label_size, activation):
        self.image_height = image_height
        self.image_width = image_width
        self.label_size = label_size
        self.activation = activation

    def weight_variable(self, shape, name=None):
        """
        创建参数w
        :param shape:
        :param name:
        :return:
        """
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def biases_variable(self, shape, name=None):
        """
        创建参数b
        :param shape:
        :param name:
        :return:
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, w, stride, name=None):
        """
        卷积层
        :param x: [batch, in_height, in_width, in_channels]
        :param w: [filter_height, filter_width, in_channels, out_channels]
        :param stride:每一步滑动的步长， strides[0]=strides[3]
        :param name:
        :return:
        """
        conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME", name=name)
        return conv

    def max_pool(self, x, stride, name=None):
        """
        pooling 层, 当 stride = ksize， padding='SAME' 时输出 tensor 大小减半
        :param x:
        :param stride:
        :param name:
        :return:
        """
        return tf.nn.max_pool(value=x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding="SAME", name=name)

    def build_model(self):
        """
        构建模型
        :return:
        """
        # 输入特征
        self.x = tf.placeholder(tf.float32, shape=[None, self.image_height*self.image_width])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.label_size])

        # 学习率
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # dropout layer : keep probability
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # reshape features to 2d shape
        self.image_x = tf.reshape(self.x, shape=[-1, self.image_height, self.image_width, 1])

        # 创建第一个卷积层的参数
        self.w_conv1 = self.weight_variable([3, 3, 1, 32])
        self.b_conv1 = self.biases_variable([32])
        # 创建第一个卷积层
        conv1 = self.activation(self.conv2d(self.image_x, self.w_conv1, stride=1, name="conv1") + self.b_conv1)
        # 创建第一个池化层
        pool1 = self.max_pool(conv1,stride=2,name='pool1')

        # 创建第二个卷积层的参数
        self.W_conv2 = self.weight_variable([3, 3, 32, 64], name='W_conv2')
        self.b_conv2 = self.biases_variable([64], name='b_conv2')
        # 创建第二个卷积层
        conv2 = self.activation(self.conv2d(pool1, self.W_conv2,stride=1, name='conv2')+ self.b_conv2)
        # 创建第二个池化层
        pool2 = self.max_pool(conv2, stride=2, name='pool2')

        # 创建第一个全连接层参数
        self.W_fc1 = self.weight_variable([7*7*64, 1024])
        self.b_fc1 = self.biases_variable([1024])
        # 第一个全连接层
        self.pool2_flat = tf.reshape(pool2, shape=[-1,7*7*64])
        fc1 = self.activation(tf.matmul(self.pool2_flat, self.W_fc1)+self.b_fc1)

        # dropout
        dropout = tf.nn.dropout(fc1,self.keep_prob)
        # 创建输出层参数
        self.W_fc2 = self.weight_variable(shape=[1024,10])
        self.b_fc2 = self.biases_variable(shape=[10])
        # 输出层
        self.y_conv = tf.matmul(dropout, self.W_fc2)+self.b_fc2
        # softmax
        self.logit_out = tf.nn.softmax(self.y_conv)

    def init_train_op(self):
        self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logit_out))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function)
        self.predict = tf.argmax(self.logit_out, 1)
        correct_prediction = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.logit_out, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def train(self, feature_x, y, learing_rate, keep_prob=0.8):
        """

        :param feature_x:
        :param y:
        :param learing_rate:
        :param keep_prob:
        :return:
        """
        feed_dict ={self.x: feature_x,
                    self.y_: y,
                    self.keep_prob: keep_prob,
                    self.learning_rate: learing_rate}
        _, loss = self.sess.run([self.train_step, self.loss_function], feed_dict=feed_dict)
        return loss
    def init(self):
        self.build_model()
        self.init_train_op()
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def classify(self,feature_x):
        """
        :param feature_x:
        :return:
        """
        feed_dict = {self.x: feature_x,
                     self.keep_prob: 1.0}

        predicts=self.sess.run(self.predict, feed_dict=feed_dict)
        return predicts

    def get_accuracy(self,x, y):
        """
        :param x:
        :param y:
        :return:
        """
        feed_dict = {self.x: x,
                     self.y_: y,
                     self.keep_prob: 1.0}
        accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
        return accuracy



def main():
    print ("load data ...")
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

    image_height = 28
    image_width = 28
    label_size = 10

    # parameters
    learning_rate = 0.1
    activation = tf.nn.relu
    training_epoches = 10
    batch_size =200
    display_set =1
    total_batch =int(mnist.train.num_examples/batch_size)
    cnn = ConvolutionNetwork(image_height=image_height, image_width=image_width, label_size=label_size, activation=activation)
    cnn.init()
    for i_epoch in range(0,training_epoches):
        avg_cost = 0
        for i_step in range(0, total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            loss = cnn.train(feature_x=batch_x, y=batch_y, learing_rate=learning_rate)
            avg_cost += loss/total_batch
        if i_epoch% display_set ==0:
            print ("Epoch:%04d, cost=:%.9f" % (i_epoch+1, avg_cost))
        if i_epoch % 4 == 0:
            learning_rate /= 2
    print ("Training finished!")
    print ('Predict....')
    accuracy = cnn.get_accuracy(mnist.test.images,mnist.test.labels)
    print ("accuracy=%.3f" % accuracy)

if __name__ == '__main__':
    main()








