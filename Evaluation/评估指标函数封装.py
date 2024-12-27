import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.cluster import rand_score
import matplotlib.pyplot as plt

class Evaluation(object):
    def calculate_f_metrics(self):
        """
        计算给定数据集中cell_type（真实类别）和e_celltype（预测类别）两列对应的F宏和F微指标，
        并将结果保存在一个数据框中返回。

        参数：
        data (pd.DataFrame): 包含cell_type和e_celltype列的数据集

        返回：
        pd.DataFrame: 包含F宏和F微指标结果的数据框
        """
        # 获取真实类别和预测类别列数据
        y_true = self['cell_type'].tolist()
        y_pred = self['e_celltype'].tolist()

        # 计算Micro-F1
        micro_f1 = f1_score(y_true, y_pred, average='micro')

        # 计算Macro-F1
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        # 创建结果数据框
        result_df = pd.DataFrame({
            'metric': ['Macro-F1', 'Micro-F1'],
            'value': [macro_f1, micro_f1]
        })
        return result_df
    
    def calculate_fmi(self):
        y_true = self['cell_type'].tolist()
        y_pred = self['e_celltype'].tolist()
        fmi = fowlkes_mallows_score(y_true, y_pred)
        result_df = pd.DataFrame({
           'metric': ['FMI'],
            'value': [fmi]
        })
        return result_df
    
    def calculate_ri(self):
        y_true = self['cell_type'].tolist()
        y_pred = self['e_celltype'].tolist()
        ri = rand_score(y_true, y_pred)
        result_df = pd.DataFrame({
           'metric': ['RI'],
            'value': [ri]
        })
        return result_df
    
    def ari(self):
        # 获取真实类别和预测类别列数据
        y_true = self['cell_type'].tolist()
        y_pred = self['e_celltype'].tolist()

        # 计算ARI
        ari = adjusted_rand_score(y_true, y_pred)

        # 创建结果数据框
        result_df = pd.DataFrame({
            'metric': 'ARI',
            'value': ari
        }, index=[0])
        return result_df
    
    def nmi(self):
        # 获取真实类别和预测类别列数据
        y_true = self['cell_type'].tolist()
        y_pred = self['e_celltype'].tolist()

        # 计算NMI
        nmi = normalized_mutual_info_score(y_true, y_pred)

        # 创建结果数据框
        result_df = pd.DataFrame({
            'metric': 'NMI',
            'value': nmi
        }, index=[0])
        return result_df
