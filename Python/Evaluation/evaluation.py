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
        Calculate the macro-F1 and micro-F1 scores for the given dataset with columns 'cell_type' (true labels) and 'e_celltype' (predicted labels), 
        and return the results in a DataFrame.

        Parameters:
        data (pd.DataFrame): Dataset containing 'cell_type' and 'e_celltype' columns

        Returns:
        pd.DataFrame: A DataFrame containing the macro-F1 and micro-F1 scores
        """
        # Retrieve true labels and predicted labels
        y_true = self['cell_type'].tolist()
        y_pred = self['e_celltype'].tolist()

        # Calculate Micro-F1
        micro_f1 = f1_score(y_true, y_pred, average='micro')

        # Calculate Macro-F1
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        # Create results DataFrame
        result_df = pd.DataFrame({
            'metric': ['Macro-F1', 'Micro-F1'],
            'value': [macro_f1, micro_f1]
        })
        return result_df
    
    def calculate_fmi(self):
        """
        Calculate the Fowlkes-Mallows Index (FMI) for the given dataset with columns 'cell_type' (true labels) and 'e_celltype' (predicted labels),
        and return the result in a DataFrame.

        Returns:
        pd.DataFrame: A DataFrame containing the FMI score
        """
        y_true = self['cell_type'].tolist()
        y_pred = self['e_celltype'].tolist()
        fmi = fowlkes_mallows_score(y_true, y_pred)
        result_df = pd.DataFrame({
           'metric': ['FMI'],
            'value': [fmi]
        })
        return result_df
    
    def calculate_ri(self):
        """
        Calculate the Rand Index (RI) for the given dataset with columns 'cell_type' (true labels) and 'e_celltype' (predicted labels),
        and return the result in a DataFrame.

        Returns:
        pd.DataFrame: A DataFrame containing the RI score
        """
        y_true = self['cell_type'].tolist()
        y_pred = self['e_celltype'].tolist()
        ri = rand_score(y_true, y_pred)
        result_df = pd.DataFrame({
           'metric': ['RI'],
            'value': [ri]
        })
        return result_df
    
    def ari(self):
        """
        Calculate the Adjusted Rand Index (ARI) for the given dataset with columns 'cell_type' (true labels) and 'e_celltype' (predicted labels),
        and return the result in a DataFrame.

        Returns:
        pd.DataFrame: A DataFrame containing the ARI score
        """
        # Retrieve true labels and predicted labels
        y_true = self['cell_type'].tolist()
        y_pred = self['e_celltype'].tolist()

        # Calculate ARI
        ari = adjusted_rand_score(y_true, y_pred)

        # Create results DataFrame
        result_df = pd.DataFrame({
            'metric': 'ARI',
            'value': ari
        }, index=[0])
        return result_df
    
    def nmi(self):
        """
        Calculate the Normalized Mutual Information (NMI) score for the given dataset with columns 'cell_type' (true labels) and 'e_celltype' (predicted labels),
        and return the result in a DataFrame.

        Returns:
        pd.DataFrame: A DataFrame containing the NMI score
        """
        # Retrieve true labels and predicted labels
        y_true = self['cell_type'].tolist()
        y_pred = self['e_celltype'].tolist()

        # Calculate NMI
        nmi = normalized_mutual_info_score(y_true, y_pred)

        # Create results DataFrame
        result_df = pd.DataFrame({
            'metric': 'NMI',
            'value': nmi
        }, index=[0])
        return result_df
