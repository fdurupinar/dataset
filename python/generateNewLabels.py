
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from scipy.stats import binom_test

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency
import numpy as np
from scipy.stats import chi2_contingency

from scipy.stats import entropy
import csv
traits = ['o', 'c', 'e', 'a', 'n']



def getNewLabels(fileIn, fileOut, actualLabels):
    df = pd.read_csv(fileIn) # results as video/trait/cnt/selectedcnt

    dfActualLabels = pd.read_csv(actualLabels)
    dfActualLabels = dfActualLabels.rename(
        columns={'File':'video', 'openness': 'o', 'conscientiousness': 'c', 'extroversion': 'e', 'agreeableness': 'a',
                 'neuroticism': 'n'})

    unique_videos = df['video'].unique()
    # Create a new DataFrame with matching rows
    new_df = dfActualLabels[dfActualLabels['video'].isin(unique_videos)]

    # Reset the index if needed
    dfNewLabels = new_df.reset_index(drop=True)

    for index, row in df.iterrows():
        video = row['video']
        trait = row['trait']
        pers = row['selectedCnt'] /row['totalCnt']

        dfNewLabels.loc[(dfNewLabels['video'] == video), trait] = pers


    dfNewLabels.to_csv(fileOut, index = False)


def createComparisonFile(fileIn, fileOut, actualLabels):
    df = pd.read_csv(fileIn)  # results as video/trait/cnt/selectedcnt

    dfActualLabels = pd.read_csv(actualLabels)
    dfActualLabels = dfActualLabels.rename(
        columns={'File': 'video', 'openness': 'o', 'conscientiousness': 'c', 'extroversion': 'e', 'agreeableness': 'a',
                 'neuroticism': 'n'})

    unique_videos = df['video'].unique()
    # Create a new DataFrame with matching rows
    # new_df = dfActualLabels[dfActualLabels['video'].isin(unique_videos)]

    # Reset the index if needed

    df['perceived'] = None
    df['actual'] = None
    for index, row in df.iterrows():
        video = row['video']
        trait = row['trait']
        pers = row['selectedCnt'] / row['totalCnt']


        df.loc[(df['video'] == video), 'perceived'] = pers
        # Extract a single value from dfActualLabels
        value_to_assign = dfActualLabels.loc[dfActualLabels['video'] == video, trait].iloc[0]

        # Assign the value to the matching rows in df
        df.loc[df['video'] == video, 'actual'] = value_to_assign


    df.to_csv(fileOut, index=False)


# getNewLabels('../mturk2/mturk2formattedResults.csv', '../mturk2/mturk2UserLabels.csv', '../mturk2/binned_regression_results.csv')

createComparisonFile('../mturk2/mturk2formattedResults.csv', '../mturk2/mturk2Confusion.csv', '../mturk2/binned_regression_results.csv')


# getNewLabels('formattedResults3.csv', 'newLabels3.csv')