import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot
from sklearn import metrics
import numpy as numpy
import pandas as pandas
import tensorflow as tf

from tensorflow.python.data import Dataset
from learning.AIConfig import AIConfig

config = AIConfig()

california_housing_dataFrame = pandas.read_csv(
    "https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataFrame = california_housing_dataFrame.reindex(
    numpy.random.permutation(california_housing_dataFrame.index))
california_housing_dataFrame["median_house_value"] /= 1000.0

# 查看数据
# california_housing_dataFrame
# california_housing_dataFrame.describe()

# 定义输入特征
my_feature = california_housing_dataFrame[["total_rooms"]]
# 为total_rooms配置一个数值特征列
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
# 定义标签
targets = california_housing_dataFrame["median_house_value"]

# 选择参数优化算法
# GradientDescent 梯度下降
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

leaner_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)
