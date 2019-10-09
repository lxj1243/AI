import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as pyplot
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
dataframe = DataImport().dataImport(
    link="https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv",
    sep=",",
    reindex=True)
dataframe["median_house_value"] /= 1000.0


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
    # 这里返回两个一样的数据可能是因为tf库里函数的要求？
    return features, labels


# 定义训练模型
def train_model(learning_rate, steps, batch_size, input_feature):
    """

    :param learning_rate: 学习速率（步长）
    :param steps: 学习的总步数
    :param batch_size: 单步的样本数量，数据集总训练样本数=batch_size*steps
    :param input_feature: 输入数据集中的输入特征（这里是单个）
    :return:
    """

    # 输出损失值的周期（步数），每10步输出一次损失值
    periods = 10
    steps_per_periods = steps / periods

    # my_feature = input_feature
    # input_feature_data = dataframe[[my_feature]].astype('float32')
    # 像上面那样写的意义是啥？
    # [[feature]] 跟[feature] 好像功效是一样的，前者print没有后缀信息，后者print有后缀信息
    input_feature_data = dataframe[[input_feature]].astype('float32')
    # 要预测的标签
    predict_label = "median_house_value"
    targets = dataframe[predict_label].astype('float32')

    # lambda 冒号前没有参数就是一个无参数匿名函数
    # 训练时的输入函数,这里定义后，之后就不用写带有一长串参数的函数再作为参数了，简洁一点
    training_input_fn = lambda: my_input_fn(input_feature_data, targets, batch_size=batch_size)
    # 预测时的输入函数
    predicing_training_input_fn = lambda: my_input_fn(input_feature_data, targets, num_epochs=1, shuffle=False)

    # 设置tensorflow特征值
    feature_columns = [tf.feature_column.numeric_column(input_feature)]
    print("feature_columns is :\n", feature_columns)

    # 创建一个线性回归object
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # contrib 包含不稳定的或实验性的代码
    # estimator.clip_gradients_by_norm 应用梯度裁剪到优化器
    # 梯度裁剪确保在训练期间梯度大小不会变得过大，过大的梯度会导致梯度下降算法失败，过大的梯度可能意味着步长设置太大了
    # 参考：https://daiwk.github.io/posts/platform-tf-basics.html ；https://blog.csdn.net/guolindonggld/article/details/79547284
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

    # 展示模型
    # 图像大小
    pyplot.figure(figsize=(15, 6))
    # 向图像添加一个子plot，参数意思是将figure分为1行2列，当前子plot在1行1列（序号顺序同阅读顺序），subplot(121)也可以表达同一个意思
    pyplot.subplot(1, 2, 1)
    pyplot.title("learned line by period")
    pyplot.ylabel(predict_label)
    pyplot.xlabel(input_feature)
    sample = dataframe.sample(n=300)
    # 散点图
    pyplot.scatter(sample[input_feature], sample[predict_label])
    colors = [cm.coolwarm(x) for x in numpy.linspace(-1, 1, periods)]

    # 在循环内训练模型，以便于周期性的评估
    print("Training model...")
    # RMSE:root mean square error
    print("RMSE(on training data):")
    # 各个周期的均方根误差组成的数组
    root_mean_squared_errors = []
    for period in range(0, periods):
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_periods)
        # 每个周期都预测一次以计算损失函数
        predictions = linear_regressor.predict(input_fn=predicing_training_input_fn)
        print(predictions)
        # predictions 里的item是二维数组？每一个item['prediction'][0]组合形成array
        predictions = numpy.array([item['predictions'][0] for item in predictions])

        # 计算损失函数——均方根误差
        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))
        print("period %02d:%0.2f" % (period, root_mean_squared_error))
        root_mean_squared_errors.append(root_mean_squared_error)
        # y轴(临时？)范围是0到预测值的最大值
        y_extents = numpy.array([0, sample[predict_label].max()])
        # 特征对应的权重值
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        # 偏差值
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        # x轴的范围：Xmin<(y-b)/w<Xmax,
        x_extents = (y_extents - bias) / weight
        x_extents = numpy.maximum(numpy.minimum(x_extents, sample[input_feature].max()), sample[input_feature].min())
        # y轴范围：wx+b
        y_extents = weight * x_extents + bias
        pyplot.plot(x_extents, y_extents, color=colors[period])
    print("model training finished")

    # 输出各个周期的损失函数矩阵图像
    pyplot.subplot(1, 2, 2)
    pyplot.ylabel('RMSE')
    pyplot.xlabel('Periods')
    pyplot.title('root mean squared error vs. periods')
    pyplot.tight_layout()
    pyplot.plot(root_mean_squared_errors)

    # 创建带有刻度数据的表
    calibration_data = pandas.DataFrame()
    calibration_data["predictions"] = pandas.Series(predictions)
    calibration_data["targets"] = pandas.Series(targets)
    display.display(calibration_data.describe())

    print("final RMSE(on training data): %0.2f" % root_mean_squared_error)

    return calibration_data


dataframe["rooms_per_person"] = dataframe["total_rooms"] / dataframe["population"]

# 未截取离群值，训练模型
calibration_data = train_model(learning_rate=0.05, steps=500, batch_size=5, input_feature="rooms_per_person")
pyplot.figure(figsize=(15,6))
pyplot.subplot(1,2,1)
pyplot.scatter(calibration_data["predictions"],calibration_data["targets"])
pyplot.subplot(1,2,2)
# 为什么要给下划线赋值？？？
_ = dataframe["rooms_per_person"].hist()

# 截取离群值
dataframe["rooms_per_person"]=dataframe["rooms_per_person"].apply(lambda x:min(x,5))
# 截取离群值后，训练模型
calibration_data = train_model(learning_rate=0.05, steps=500, batch_size=5, input_feature="rooms_per_person")
pyplot.figure(figsize=(15,6))
pyplot.subplot(1,2,1)
pyplot.scatter(calibration_data["predictions"],calibration_data["targets"])
pyplot.subplot(1,2,2)
_ = dataframe["rooms_per_person"].hist()