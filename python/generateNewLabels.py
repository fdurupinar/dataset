
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score


import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency
import numpy as np
from scipy.stats import chi2_contingency

from scipy.stats import entropy
import csv
traits = ['o', 'c', 'e', 'a', 'n']

import pandas as pd
from collections import defaultdict

def calculateSelectionCountPerVideo(dfResponses):
    # Rename columns for easier access
    dfResponses = dfResponses.rename(columns={
        'Answer.hO.left': 'o.left', 'Answer.hO.right': 'o.right', 'Answer.hO.equal': 'o.equal',
        'Answer.hC.left': 'c.left', 'Answer.hC.right': 'c.right', 'Answer.hC.equal': 'c.equal',
        'Answer.hE.left': 'e.left', 'Answer.hE.right': 'e.right', 'Answer.hE.equal': 'e.equal',
        'Answer.hA.left': 'a.left', 'Answer.hA.right': 'a.right', 'Answer.hA.equal': 'a.equal',
        'Answer.hN.left': 'n.left', 'Answer.hN.right': 'n.right', 'Answer.hN.equal': 'n.equal',
        'Input.openness_video1': 'o.video.left', 'Input.openness_video2': 'o.video.right',
        'Input.conscientiousness_video1': 'c.video.left', 'Input.conscientiousness_video2': 'c.video.right',
        'Input.extroversion_video1': 'e.video.left', 'Input.extroversion_video2': 'e.video.right',
        'Input.agreeableness_video1': 'a.video.left', 'Input.agreeableness_video2': 'a.video.right',
        'Input.neuroticism_video1': 'n.video.left', 'Input.neuroticism_video2': 'n.video.right'
    })


    traits = ['o', 'c', 'e', 'a', 'n']
    selected_cnt = defaultdict(int)
    total_cnt = defaultdict(int)
    new_rows = []
    for _, row in dfResponses.iterrows():
        for trait in traits:
            left_file = row[f'{trait}.video.left'].rsplit('/', 1)[-1]
            right_file = row[f'{trait}.video.right'].rsplit('/', 1)[-1]

            # Update total and selected counts
            # new_rows.append({'video': left_file, 'trait': trait, 'selectedCnt': int(row[f'{trait}.left']), 'totalCnt': 1})
            # new_rows.append({'video': right_file, 'trait': trait, 'selectedCnt': int(row[f'{trait}.right']), 'totalCnt': 1})

            if row[f'{trait}.left'] == True:
                selected_cnt[(left_file, trait)] += 1
            else:
                selected_cnt[(right_file, trait)] += 1
            total_cnt[(left_file, trait)] +=1
            total_cnt[(right_file, trait)] += 1

    df = pd.DataFrame([
        {'video': video, 'trait':trait, 'selectedCnt': selected_cnt[(video,trait)], 'totalCnt': total_cnt[(video,trait)]}
        for video, trait in total_cnt.keys()
    ])

    # Create the final DataFrame
    # df = pd.DataFrame(new_rows)
    # df = df.groupby(['video', 'trait']).sum().reset_index()
    #
    # print("Correct Counts:", dict(selected_cnt))
    # print("Incorrect Counts:", dict(incorrectCnt))
    # print("Total Counts:", dict(total_cnt))
    # print("Accuracy:", accuracy)

    return df



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

def get_video_labels(fileIn, fileOut):
    df = pd.read_csv(fileIn)  # results as video/trait/cnt/selectedcnt
    # Create an empty DataFrame for the desired output
    output_rows = []

    # Iterate through each row of the input file
    for _, row in df.iterrows():
        # Extract the relevant data for each trait
        traits = ['extroversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

        for i in [1, 2]:  # For both "video1" and "video2"
            # Extract file name (last part of URL)
            file_name = row[f'{traits[0]}_video{i}'].rsplit('/', 1)[-1]

            # Extract trait values for the corresponding file
            trait_values = [row[f'{trait}_value{i}'] for trait in traits]

            # Append the processed row to the output
            output_rows.append([file_name] + trait_values)

    # Create a DataFrame for the output
    output_df = pd.DataFrame(output_rows, columns=['File'] + traits)

    # Write the output DataFrame to a CSV file
    output_df.to_csv(fileOut, index=False)

# getNewLabels('../mturk2/mturk2formattedResults.csv', '../mturk2/mturk2UserLabelsIncorrect.csv', '../mturk2/binned_regression_results.csv')

# createComparisonFile('../mturk2/mturk2formattedResults.csv', '../mturk2/mturk2Confusion.csv', '../mturk2/binned_regression_results.csv')


# getNewLabels('formattedResults3.csv', 'newLabels3.csv')

dfResponses = pd.read_csv('../mturk2/mturk2ValidResults.csv')
dfFormatted = calculateSelectionCountPerVideo(dfResponses)
dfFormatted.to_csv("../mturk2/mturk2formattedResults.csv", index=False)
getNewLabels("../mturk2/mturk2formattedResults.csv", "../mturk2/mturk2UserLabels.csv", '../mturk2/binned_regression_results.csv')

# get_video_labels('../mturk3/mturk3_input_fixed.csv', '../mturk3/mturk3RegressionLabels.csv')
