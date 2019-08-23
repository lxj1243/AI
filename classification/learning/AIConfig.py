import tensorflow as tensorflow
import pandas as pandas


class AIConfig:
    # 设置过后，整个运行周期内都会采用这种配置，不用返回设置过的对象
    def __init__(self,max_rows,max_columns):
        tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
        pandas.options.display.max_rows = max_rows;
        pandas.options.display.max_columns = max_columns;
        pandas.options.display.float_format = '{:.1f}'.format
