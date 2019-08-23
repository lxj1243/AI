import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib as matplotlib
import numpy as numpy
import pandas as pandas
import sklearn.metrics as metrics
import tensorflow as tf

from tensorflow.python.data import Dataset
# from learning import AiConfig,DataImport 不行...
from learning.AIConfig import AIConfig
from learning.DataImport import DataImport

# 实例化对象时自动调用__init__函数
config = AIConfig(None, None)
dataframe = DataImport().dataImport("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", ",")
# print(dataframe)


# 定义输入函数
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    :param features: 输入数据（特征与特征值）
    :param batch_size:
    :param shuffle:
    :param num_epochs:
    :return:
    """

    # dict(features).items()在输入数据上建立（key，value）迭代器，value是一列值（series）
    # <print打印value出来还有index和统计信息，但是array方法用过后这些就print不出来了>
    # 对于每一个（key，value） numpy.array(value）把value转换成数组，（key，value<array>）
    features = {key: numpy.array(value) for key, value in dict(features).items()}
    # 构建数据集，tensor是TensorFlow程序中的主要数据结构。张量是N维数据结构，最常见的是标量、向量或矩阵。
    # 张量的元素可以包含整数值、浮点值或字符串值。
    # from_tensor_slices使用给出的张量（就是features and targets）创建数据集，2GB limit
    dataset = Dataset.from_tensor_slices((features, targets))
    # batch是模型训练的一次迭代（即一次梯度更新）中使用的样本集
    # num_epochs：数据集重复的次数
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    # 对数据进行洗牌，随机从数据缓存区里取出数据，对数据进行重排
    if shuffle: dataset = dataset.shuffle(buffer_size=10000)
    # 在数据集上建立一个迭代器，one_shot意味着单次的迭代器，不支持动态的数据集，不支持参数化。
    # 什么是参数化:可以理解为单次的 Iterator 需要 Dataset 在程序运行之前就确认自己的大小，
    # Tensorflow中有一种feeding机制,它允许程序运行时再真正决定需要的数据，单次的 Iterator 不能满足这种要求
    # 可以使用 make_initialisable_iterator()
    # 版权声明：本文为CSDN博主「frank909」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/briblue/article/details/80962728
    # return下一个batch
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels


# 定义训练模型
def train_model(learning_rate, steps, batch_size, input_feature):
    """

    :param learning_rate: 学习速率（步长）
    :param steps: 学习的总步数
    :param batch_size: 单步的样本数量，数据集总训练样本数=batch_size*steps
    :param input_feature: 输入数据集中的输入特征（单个）
    :return:
    """

    # 输出损失值的周期（步数），每10步输出一次损失值
    periods = 10
    steps_per_periods = steps / periods

    # my_feature = input_feature
    # my_feature_data = dataframe[[my_feature]].astype('float32')
    # 像上面那样写的意义是啥？
    input_feature_data = dataframe[[input_feature]].astype('float32')
    input_label = "median_house_value"
    targets = dataframe[input_label].astype('float32')

    # lambda 冒号前没有参数就是一个无参数匿名函数
    # 训练时的输入函数
    training_input_fn = lambda: my_input_fn(input_feature_data, targets, batch_size=batch_size)
    # 预测时的输入函数
    predicing_training_input_fn = lambda: my_input_fn(input_feature_data, targets, num_epochs=1, shuffle=False)

    # 设置tensorflow特征值
    feature_columns = [tf.feature_column.numeric_column(input_feature)]
    print("feature_columns is :\n", feature_columns)

    # 创建一个线性回归object
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,optimizer=my_optimizer)

