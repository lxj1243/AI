import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

from learning.AIConfig import AIConfig
from learning.DataImport import DataImport

config = AIConfig(None, None)
# ca_house_dataframe = DataImport.dataImport(
#      "https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv",
#      ",")   这样不对
ca_house_dataframe = DataImport().dataImport(
    link="https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv",
    sep=",",
    reindex=False)


# 数据预处理，通过计算生成一些新的特征等
# 这里把特征名都写死在方法里的原因大概是：不同的数据集或者不同的数据使用目的预处理方法将会完全不同
def ca_house_preprocess_features(ca_house_dataframe):
    """

    :param ca_house_dataframe:
    :return:
    """
    # seleted_features = ca_house_dataframe[[
    ca_house_processed_features = ca_house_dataframe[[
        "latitude",
        "longitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"]]
    # 下面这一句的意义是什么？https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html
    # processed_features = seleted_features.copy()
    ca_house_processed_features["rooms_per_person"] = ca_house_dataframe["total_rooms"] / ca_house_dataframe["population"]
    # 转换格式，后面好处理一些？原本的dataframe就是一个多维矩阵（像数据库的表，有表头和行索引），
    # dict转化之后变成了表头（key）：特征数据（value）形式的迭代器（只在key上迭代），items方法后value也迭代
    # 在特征和特征数据上建立迭代器后，将其再转换为特征（key）：特征数据（数组）的形式，可能是因为tf里的方法要求用这样的数据？
    ca_house_processed_features = {key: np.array(value) for key, value in dict(ca_house_processed_features).items()}
    return ca_house_processed_features


# 选取并处理一个特征作为标签
def ca_house_preprocess_targets(ca_house_dataframe):
    """

    :param ca_house_dataframe:
    :return:
    """
    output_targets = pd.DataFrame()
    output_targets["median_house_value"] = ca_house_dataframe["median_house_value"] / 1000.0
    return output_targets


# 回归/分类等算法的输入函数，包括数据、标签、步长等
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """

    :param features:
    :param targets:
    :param batch_size:
    :param shuffle:
    :param num_epochs:
    :return:
    """

    # 用tf的方法处理数据，在数据上设置一次梯度下降验证时所用的数据集大小（batch_size),以及数据集重复的次数（num_epchs)
    # 具体怎么整的不清楚，得看tf的源码了吧
    dataset = Dataset.from_tensor_slices(features, targets)
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels


# 由于我们现在使用的是多个输入特征，因此需要把用于将特征列配置为独立函数的代码模块化:设定特征的类型<文本或数字>？
# 目前此代码相当简单，因为我们的所有特征都是数值，但当我们在今后的练习中使用其他类型的特征时，会基于此代码进行构建。
def construct_feature_columns(input_features):
    """

    :param input_features:
    :return:
    """
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """

    :param learning_rate:
    :param steps:算这么多次损失值<均方误差>之后，训练样本就用完了
    :param batch_size:每次取这么多个样本计算损失值<均方误差>
    :param training_examples:
    :param training_targets:
    :param validation_examples:
    :param validation_targets:
    :return:
    """
    # 每下降（优化）10次，计算一次损失值
    # periods = 10
    # steps_per_period = steps / periods

    # 回归器需要先设定好优化器和特征，然后根据输入函数周期性优化、计算损失值
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(construct_feature_columns(training_examples),my_optimizer)

    input_fn = lambda : my_input_fn(training_examples,training_targets,batch_size)

# 训练集
training_examples = preprocess_features(dataframe.head(12000))
training_targets = preprocess_targets(dataframe.head(12000))

# 验证集
validation_examples = preprocess_features(dataframe.tail(5000))
validation_targets = preprocess_targets(dataframe.tail(5000))

# matplotlib的很多函数都没法自动补充是咋回事？？
# plt.figure(figsize=(13, 8))
#
# splt = plt.subplot(1, 2, 1)
# splt.set_title("Validation Data")
#
# splt.set_autoscalex_on(False)
# splt.set_ylim([32, 43])
# splt.set_autoscaley_on(False)
# splt.set_xlim([-126, -112])
# plt.scatter(validation_examples["longitude"],
#             validation_examples["latitude"],
#             c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())
#
# ax = plt.subplot(1, 2, 2)
# ax.set_title("Training Data")
#
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# plt.scatter(training_examples["longitude"],
#             training_examples["latitude"],
#             c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
# _ = plt.plot()
