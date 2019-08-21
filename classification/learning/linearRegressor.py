from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot
from sklearn import metrics
import numpy as numpy
import pandas as pandas
import tensorflow as tensorflow

from tensorflow.python.data import Dataset

tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
pandas.options.display.max_rows=10;
pandas.options.display.float_format='{:.1f}'.format

california_housing_dataframe = pandas.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv",sep = ",")
california_housing_dataframe = california_housing_dataframe.reindex(numpy.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0

#查看数据
#california_housing_dataframe
#california_housing_dataframe.describe()

#定义输入特征
my_feature = california_housing_dataframe[["total_rooms"]]
#为total_rooms配置一个数值特征列
feature_columns = [tensorflow.feature_column.numeric_column("total_rooms")]
#定义标签
targets = california_housing_dataframe["median_house_value"]

#选择参数优化算法
#GradientDescent 梯度下降
my_optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tensorflow.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)

leaner_regressor = tensorflow.estimator.LinearRegressor(feature_columns=feature_columns,optimizer=my_optimizer)
