import pandas as pandas

print(pandas.__version__)

name = pandas.Series(['ZhangSan', 'WangWu', 'LiSi'])
age = pandas.Series(['20', '21', '22'])
people = pandas.DataFrame({'name': name, 'age': age})

california_housing_dataFrame = pandas.read_csv("https://download.mlcc.google.cn/mledu-datasets"
                                               "/california_housing_train.csv", sep=',')
# print pandas 数据行列天多，显示不全
# pandas.set_option 可以进行显示格式设置
# 行列不限制，注意是None不是none
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
# 设置特征值显示长度, 默认值为50
pandas.set_option('max_colwidth',100)
# python字符串组合直接逗号就行，不用+号
print("describe:\n",california_housing_dataFrame.describe())
print("\nhead:\n",california_housing_dataFrame.head())
hist=california_housing_dataFrame.hist('housing_median_age')
