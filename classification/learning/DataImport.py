import pandas as pandas
import numpy as numpy


class DataImport():
    def dataImport(self, link, sep, reindex):
        dataframe = pandas.read_csv(
            link, sep=sep)
        if reindex:
            dataframe = dataframe.reindex(numpy.random.permutation(dataframe.index))
        return dataframe
