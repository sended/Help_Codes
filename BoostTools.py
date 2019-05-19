# -*- coding: utf-8 -*-
"""
Created on Fri May 17 23:14:31 2019

@author: NitroXquantic
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy
import pandas
from enum import Enum
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error,  r2_score


class Importance:
    
    def __init__(self):
        pass
    
       
    def see_inportance(self, X, y, isRegressor = False):
        if isRegressor:
            return self.__regressor(X, y)
        return self.__classify(X, y)
        
    def __classify(self, X, y):
        X['random'] = numpy.random.rand(len(X))
        DTC = DecisionTreeClassifier(criterion='gini')
        DTC.fit(X, y)
        columns = X.columns
        gini = DTC.feature_importances_
        DTC = DecisionTreeClassifier(criterion='entropy')
        DTC.fit(X, y)
        entropy = DTC.feature_importances_
        table = []
        for i in range(len(columns)):
            table.append([columns[i], numpy.round(gini[i] * 100, decimals=2), numpy.round(entropy[i] * 100, decimals=2)])
        table = numpy.array(table)
        del X['random']
        return pandas.DataFrame(table, columns = ['Atribute', '% gini', '% entropy'])
    
    def __regressor(self, X, y):
        X['random'] = numpy.random.rand(len(X))
        DTR = DecisionTreeRegressor(criterion='mse')
        DTR.fit(X, y)
        columns = X.columns
        mse = DTR.feature_importances_
        DTR = DecisionTreeRegressor(criterion='mae')
        DTR.fit(X, y)
        mae = DTR.feature_importances_
        table = []
        for i in range(len(columns)):
            table.append([columns[i], numpy.round(mse[i] * 100, decimals=2), numpy.round(mae[i] * 100, decimals=2)])
        table = numpy.array(table)
        del X['random']
        return pandas.DataFrame(table, columns = ['Atribute', '% mse', '% mae'])


class Scorings(Enum):
    #Classificação
    Accuracy = 'accuracy'
    Balanced_accuracy = 'balanced_accuracy'
    Average_precision = 'average_precision'
    Brier_score_loss = 'brier_score_loss'
    F1 = 'f1'
    F1_micro = 'f1_micro'
    F1_macro = 'f1_macro'
    F1_weighted = 'f1_weighted'
    F1_samples = 'f1_samples'
    Neg_log_loss = 'neg_log_loss'
    Precision = 'precision'
    Recall = 'recall'
    Roc_auc = 'roc_auc'
    #Clusterização
    Adjusted_mutual_info_score = 'adjusted_mutual_info_score'
    Adjusted_rand_score = 'adjusted_rand_score'
    Completeness_score = 'completeness_score'
    Fowlkes_mallows_score = 'fowlkes_mallows_score'
    Homogeneity_score = 'homogeneity_score'
    Mutual_info_score = 'mutual_info_score'
    Normalized_mutual_info_score = 'normalized_mutual_info_score'
    V_measure_score = 'v_measure_score'
    #Regressão
    Explained_variance = 'explained_variance'
    Neg_mean_absolute_error = 'neg_mean_absolute_error'
    Neg_mean_squared_error = 'neg_mean_squared_error'
    Neg_mean_squared_log_error = 'neg_mean_squared_log_error'
    Neg_median_absolute_error = 'neg_median_absolute_error'
    R2 = 'r2'
    
class CrossValidation:
    
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model
        
    
    def KFoldValidator(self, scoring, number_of_splits = 4, random = False, random_seed = None, show_each_test = False):
        kfolf = KFold(n_splits=number_of_splits, shuffle=random, random_state=random_seed)
        cross_result = cross_validate(self.model, self.X, self.y, cv=kfolf, n_jobs=-1, scoring=scoring.value)
        train_score = cross_result['train_score']
        test_score = cross_result['test_score']
        mean_train = numpy.mean(train_score)
        sd_train = numpy.std(train_score)
        mean_test = numpy.mean(test_score)
        sd_test = numpy.std(test_score)
        print('Resultado em %s' % (scoring.value))
        print('-'*100)
        print('A média do teste é %.2f%% com desvio padrão de %.2f%%' % ((mean_test * 100.0), (sd_test * 100.0)))
        print('A média do treino é %.2f%% com desvio padrão de %.2f%%' % ((mean_train * 100.0), (sd_train * 100.0)))
        if show_each_test:
            print('*'*100)
            print('Cada treino')
            for i, number in enumerate(train_score):
                print('treino %d com %.2f%%' % ((i + 1), (number * 100.0)))
            print('-'*100)
            for i, number in enumerate(test_score):
                print('teste %d com %.2f%%' % ((i + 1), (number * 100.0)))
        
    def LeaveOneOutValidator(self, scoring):
        leaveOneOut = LeaveOneOut()
        cross_result = cross_validate(self.model, self.X, self.y, cv=leaveOneOut, scoring=scoring.value, n_jobs=-1)
        train_score = cross_result['train_score']
        test_score = cross_result['test_score']
        mean_train = numpy.mean(train_score)
        sd_train = numpy.std(train_score)
        max_train = numpy.max(train_score)
        min_train = numpy.min(train_score)
        mean_test = numpy.mean(test_score)
        sd_test = numpy.std(test_score)
        max_test = numpy.max(test_score)
        min_test = numpy.min(test_score)
        print('Resultado em %s' % (scoring.value))
        print('-'*100)
        print('A média do teste é %.2f%% com desvio padrão de %.2f%%' % ((mean_test * 100.0), (sd_test * 100.0)))
        print('O minimo de acerto é %.2f%% e o máximo é %.2f%%' % ((min_test * 100.0), (max_test * 100.0)))
        print('-'*100)
        print('A média do treino é %.2f%% com desvio padrão de %.2f%%' % ((mean_train * 100.0), (sd_train * 100.0)))
        print('O minimo de acerto é %.2f%% e o máximo é %.2f%%' % ((min_train * 100.0), (max_train * 100.0)))
        
    def HoldOutValidator(self, porcentual_test, isCategorical):
        trX = None
        tsX = None
        trY = None
        tsY = None
        if isCategorical:
            trX, tsX, trY, tsY = train_test_split(self.X, self.y, test_size = porcentual_test, stratify = self.y)
        else:
            trX, tsX, trY, tsY = train_test_split(self.X, self.y, test_size = porcentual_test)
        self.model.fit(trX, trY)
        pred = self.model.predict(tsX)
        
        if isCategorical:
            print(confusion_matrix(tsY, pred))
            print(classification_report(tsY, pred))
        else:
            print('mean_absolute_error')
            print(mean_absolute_error(tsY, pred))
            print('-'*100)
            print('r2_score')
            print(r2_score(tsY, pred))