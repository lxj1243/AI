import numpy as numpy
import pandas as pandas
import tensorflow as tf

from learning.DataImport import DataImport

tf.logging.set_verbosity(tf.logging.ERROR)
pandas.options.display.max_rows = 20
pandas.options.display.max_columns = None
pandas.options.display.float_format = '{:.1f}'.format

dataframe = pandas.read_csv(
            "https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")
# dataframe = dataframe.reindex(numpy.random.permutation(dataframe.index))
dataframe["median_house_value"] /= 1000.0
print(dataframe)
# 花括号代表字典数据类型 dict = {'jon':'boy','lili"':'girl'}
# 以下两个featuredict有微妙的区别，第一个只在key上有迭代，第二个同时以key和value迭代
# https://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops/3294899#3294899
featuredict = dict(dataframe)
# print(featuredict)
featuredict = dict(dataframe).items()
# print(featuredict)
#for key, value in featuredict:
  #  print("key is:",key,"\n","value is:",value,"value end")
# 谜之赋值方式，遍历字典类型数据featuredict中的键值对，对值（value）应用numpy的生成数组方法
data = {key: numpy.array(value) for key, value in featuredict}
#print(data)
