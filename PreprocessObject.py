# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

class PreprocessObject:
    
    __dataOriginal = 'Natural'
    
    def __init__(self, preprocessObject = None):
        
        if preprocessObject is not None:
            if type(preprocessObject) == PreprocessObject:
                self.columnsName = preprocessObject.GetColumnsName()
                self.dataframe = preprocessObject.GetDataFrame()
                self.hasPreprocess = preprocessObject.hasPreprocess
                if self.columnsName is not None:
                    self.__create_preprocess_dic()
            else:
                self.hasPreprocess = False
        else:
            self.hasPreprocess = False
    
    def convertInPreprocessObject(self, dataset):
        assert type(dataset) == pd.core.frame.DataFrame
        colummsName = dataset.columns.values
        dataframe = {}
        
        for name in colummsName:
            dataframe[name] = dataset[name].values
        
        self.columnsName = colummsName
        self.dataframe = dataframe
        self.hasPreprocess = True
        self.__create_preprocess_dic()
    
    def GetColumnsName(self):
        if self.hasPreprocess:
            return self.columnsName
        return None
    
    def GetDataFrame(self):
        if self.hasPreprocess:
            return self.dataframe
        return None
    
    
    def convertInPandasDataframe(self, this = True, preprocessObject = None):
        if not this:
            assert preprocessObject is not None, 'PreprocessObject can not None'
               
        if this:
            return pd.DataFrame(self.dataframe)
        return pd.DataFrame(preprocessObject.dataframe)
    
    def removeIten(self, name):
        assert name in self.columnsName, 'Do not have name in dataFrame'
        self.dataframe.pop(name)
        self.columnsName = np.array(list(self.dataframe.keys()))
        self.pp_dic.pop(name)
        
    def addIten(self, name, array):
                       
        if type(array) == list:
            array = np.array(array)
        elif type(array) == pd.core.series.Series:
            array = array.values
        elif type(array) == tuple:
            array = np.array(array)
        elif type(array) != np.ndarray:
            assert False, 'The type Array must be "List", "Tuple","pandas.core.frame.Dataframe" or "numpy.ndarray"'
        
        size = 0
        for n in self.dataframe:
            size = len(self.dataframe[n])
            break
        
        assert size == len(array), 'The size of the array must be equal to that of the column'
        
        self.dataframe[name] = array        
        self.columnsName = np.array(list(self.dataframe.keys()))
        self.pp_dic[name] = self.__dataOriginal #'Natural'
    
    def reorder(self, order_numbers):
        st = set(order_numbers)
        assert len(order_numbers) == len(st), "Order can't have repeated numbers"
        
        columns = list(self.dataframe.keys())
        len_columns = len(columns)
        
        if len(order_numbers) != len_columns:
            rest = []
            for number in range(len_columns):
                if number not in order_numbers:
                    rest.append(number)
            
            for number in rest:
                order_numbers.append(number)
            
        reordened_dataFrame = {}
        self.columnsName = []
        
        for number in order_numbers:
            name = columns[number]
            reordened_dataFrame[name] = self.dataframe[name]
            self.columnsName.append(name)
        
        self.columnsName = np.array(self.columnsName)
        self.dataframe = reordened_dataFrame
        
    def __create_preprocess_dic(self):
        self.pp_dic = {}
        for name in self.columnsName:
            self.pp_dic[name] = self.__dataOriginal #'Natural'
            
    def encode(self, names):
       isMultiple = self.__verify_names(names)
       if isMultiple:
           for name in names:
               label_encode = LabelEncoder()
               original = self.dataframe[name]
               data_encode = original.copy()
               data_encode = label_encode.fit_transform(data_encode)
               self.dataframe[name] = data_encode
               self.pp_dic[name] = ('LabelEncode', label_encode, original)
       else:
           label_encode = LabelEncoder()
           original = self.dataframe[names]
           data_encode = original.copy()
           data_encode = label_encode.fit_transform(data_encode)
           self.dataframe[names] = data_encode
           self.pp_dic[names] = ('LabelEncode', label_encode, original)
           
    def decode(self, name, numbers):
        _ = self.__verify_names(name)
        if type(numbers) == int:
           numbers = [numbers]         
        
        if type(self.pp_dic[name]) == tuple and self.pp_dic[name][0] == 'LabelEncode':
            label_endcode = self.pp_dic[name][1]
            return label_endcode.inverse_transform(numbers)
        return None
    
    def restaure_column(self, names):
        isMultiple = self.__verify_names(names)
        
        if isMultiple:
            for name in names:
                if type(self.pp_dic[name]) == tuple:
                    if self.pp_dic[name][0] == 'LabelEncode' or self.pp_dic[name][0] == 'StandardScaler':
                        self.dataframe[name] = self.pp_dic[name][2]
                        self.pp_dic[name] = self.__dataOriginal
                    elif self.pp_dic[name][0] == 'OnHotEncode':
                        for cols in self.pp_dic[name][1]:
                            self.dataframe.pop(cols)
                        self.dataframe[name] = self.pp_dic[name][2]
                        self.pp_dic[name] = self.__dataOriginal
        else:
            if type(self.pp_dic[names]) == tuple:
                if self.pp_dic[names][0] == 'LabelEncode' or self.pp_dic[names][0] == 'StandardScaler':
                    self.dataframe[names] = self.pp_dic[names][2]
                    self.pp_dic[names] = self.__dataOriginal
                elif self.pp_dic[names][0] == 'OnHotEncode':
                    for cols in self.pp_dic[names][1]:
                        self.dataframe.pop(cols)
                    self.dataframe[names] = self.pp_dic[names][2]
                    self.pp_dic[names] = self.__dataOriginal
                    
    def scaler(self, names):
        isMultiple = self.__verify_names(names)
        if isMultiple:
           for name in names:
               scale = StandardScaler()
               original = self.dataframe[name]
               data_scale = original.copy()
               data_scale = data_scale.reshape(-1, 1)
               data_scale = scale.fit_transform(data_scale)
               self.dataframe[name] = data_scale.reshape(1, -1)[0]
               self.pp_dic[name] = ('StandardScaler', scale, original)
        else:
           scale = StandardScaler()
           original = self.dataframe[names]
           data_scale = original.copy()
           data_scale = data_scale.reshape(-1, 1)
           data_scale = scale.fit_transform(data_scale)
           self.dataframe[names] = data_scale.reshape(1, -1)[0]
           self.pp_dic[names] = ('StandardScaler', scale, original)
    
    def unscaler(self, name, numbers):
        _ = self.__verify_names(name)
       
        if type(numbers) == tuple or type(numbers) == list:
            numbers = np.array(numbers)
        else:
            numbers = np.array([numbers])
        
        numbers = numbers.reshape(-1, 1)
                
        if type(self.pp_dic[name]) == tuple and self.pp_dic[name][0] == 'StandardScaler':
            standerdScaler = self.pp_dic[name][1]
            return standerdScaler.inverse_transform(numbers).reshape(1, -1)[0]
        return None
            
     
    def hot_encode(self, names):
        isMultiple = self.__verify_names(names)
        if isMultiple:
            for name in names:
                original = self.dataframe[name]
                for_encode = original.copy()
                for_encode = for_encode.reshape(-1, 1)
                hot_enc = OneHotEncoder(sparse=False)
                hot_enc.fit(for_encode)
                on_hot_table = hot_enc.transform(for_encode)
                names_cols = hot_enc.categories_[0]
                self.pp_dic[name] = ('OnHotEncode', tuple(names_cols), original)
                self.dataframe.pop(name)
                for i, n in enumerate(names_cols):
                    self.dataframe[n] = on_hot_table[:,i]
        else:
            original = self.dataframe[names]
            for_encode = original.copy()
            for_encode = for_encode.reshape(-1, 1)
            hot_enc = OneHotEncoder(sparse=False)
            hot_enc.fit(for_encode)
            on_hot_table = hot_enc.transform(for_encode)
            names_cols = hot_enc.categories_[0]
            self.pp_dic[names] = ('OnHotEncode', tuple(names_cols), original)
            self.dataframe.pop(names)
            for i, n in enumerate(names_cols):
                self.dataframe[n] = on_hot_table[:,i]
    
    def __verify_names(self, names):
        if type(names) == str:
            assert names in self.columnsName, "Do not found name in dataframe"
            return False            
            
        elif type(names) == list or type(names) == tuple or type(names) == np.ndarray:
            for name in names:
                assert name in self.columnsName, "Do not found name in dataframe"
            return True
        else:
            assert False, "names must be 'str', 'list', 'tuple' or 'numpy.ndarray'"
        