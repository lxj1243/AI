import pandas as pandas
import numpy as numpy


class DataImport():
    def dataImport(self, link, sep):
        dataframe = pandas.read_csv(
            link, sep=sep)
        dataframe = dataframe.reindex(
            numpy.random.permutation(dataframe.index))
        return dataframe
