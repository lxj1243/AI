import matplotlib as matplotlib
import pandas as pandas
import numpy as numpy

print(pandas.__version__)

name = pandas.Series(['ZhangSan', 'WangWu', 'LiSi'])
# 使用数字时注意类型，str类型 age = pandas.Series(['20', '21', '22'])
age = pandas.Series([20, 21, 22])
people = pandas.DataFrame({'names': name, 'ages': age})

california_housing_dataFrame = pandas.read_csv("https://download.mlcc.google.cn/mledu-datasets"
                                               "/california_housing_train.csv", sep=',')
# print pandas 数据行列太多时会显示不全
# pandas.set_option 可以进行显示格式设置
# 行列不限制，注意是None不是none
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
# 设置特征值显示长度, 默认值为50
pandas.set_option('max_colwidth', 100)
# python字符串组合直接逗号就行，不用+号
print("describe:\n", california_housing_dataFrame.describe())
print("\nhead:\n", california_housing_dataFrame.head())
# 获取绘图后台使用的库
print(matplotlib.get_backend())
# 修改后台绘图库的方法：https://vra.github.io/2017/06/13/mpl-backend/  有些不起作用
# 默认绘图库有问题，主动设定使用Qt5Agg就能显示出图像，有人说TkAgg也可以，但是这里不行,应该是缺包
matplotlib.use('Qt5Agg')
# 画图
california_housing_dataFrame.hist('housing_median_age')

print(type(people['names']))
print(people['names'][0:2])
print(type(people[0:2]))
print(people[0:2])

# numpy 运算
print(age / 10)

# lambda 函数是映射函数，age中每一个值都会应用冒号后的函数
isBigAge = age.apply(lambda eachAge: eachAge > 21)
print(isBigAge)

people['gender'] = pandas.Series(['男', '女'])
people['salary'] = pandas.Series([1000, 2000, 1500])
people['income'] = people['ages'] * people['salary']
# 这样不行：people['height'] = people['age']/10+1.6
print(people)

# exercise 1
# 错的:isTrue = people.apply(lambda eachPerson:eachPerson['ages']>21 & eachPerson['salary']>1400)
# 注意括号
people['isTrue'] = people['ages'].apply(lambda eachAge: eachAge > 20) & (people['salary'] > 1400)
print(people)

# 构造数据时，pandas会生成index，index创建后是稳定的，不会因为数据排序而改变
print(people['names'].index)
print(people.index)
# 可以主动改变index
# reindex不会给people赋值，要主动赋值
people = people.reindex([2, 0, 1])
print('after reindex:\n', people)
# 随机重排
people = people.reindex(numpy.random.permutation(people.index))
print(people)

# exercise 2
# 数据不规整有丢失的索引时，不必担心输入被清理?
people = people.reindex([0, 4, 2, 6])
print(people)
