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
# dataframe = DataImport.dataImport("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", ",")   这样不对
dataframe = DataImport().dataImport(
    link="https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv",
    sep=",",
    reindex=False)


def preprocess_features(dataframe):
    """

    :param dataframe:
    :return:
    """
    # seleted_features = dataframe[[
    processed_features = dataframe[[
        "latitude",
        "longitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"]]
    # 下面这一句的意义是什么？
    # processed_features = seleted_features.copy()
    processed_features["rooms_per_person"] = dataframe["total_rooms"] / dataframe["population"]
    return processed_features


def preprocess_targets(dataframe):
    """

    :param dataframe:
    :return:
    """
    output_targets = pd.DataFrame()
    output_targets["median_house_value"] = dataframe["median_house_value"] / 1000.0
    return output_targets


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """

    :param features:
    :param targets:
    :param batch_size:
    :param shuffle:
    :param num_epochs:
    :return:
    """

    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices(features,targets)
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features,labels = ds.make_one_shot_iterator().get_next()
    return features,labels


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
    :param steps:
    :param batch_size:
    :param training_examples:
    :param training_targets:
    :param validation_examples:
    :param validation_targets:
    :return:
    """
    periods=10
    steps_per_period=steps/periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)


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
