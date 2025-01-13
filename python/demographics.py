
import os
import pandas as pd
from statsmodels.stats.multitest import multipletests

from scipy.stats import chi2_contingency

dfInput = pd.read_csv('../csv/formattedResults3.csv')
columns_of_interest = [
    'video.left', 'video.right'
]

# Flatten all unique values from the specified columns into a single set
unique_names = set()
for col in columns_of_interest:
    if col in dfInput.columns:
        unique_names.update(dfInput[col].dropna().unique())

# Count the number of unique names
unique_videos_total = len(unique_names)



#
# # Count unique videos in video.left and video.right columns
# unique_videos_left = dfInput['video.left'].nunique()
# unique_videos_right = dfInput['video.right'].nunique()
#
# # Combine unique counts (considering the possibility of overlap between left and right videos)
# unique_videos_total = pd.concat([dfInput['video.left'], dfInput['video.right']]).nunique()

print(f"Total number of unique videos: {unique_videos_total}")


print(f"Total number of unique workers: {dfInput['workerId'].nunique()}")


def print_demographics(name):


    df = pd.read_csv(name)
    df2 = pd.DataFrame(columns=['pRace', 'pGender', 'pAge', 'pId'])


    pRace = []
    pGender = []
    pId = []
    pAge = []

    df = df.rename(columns={'WorkerId': 'pId'})

    df = df.rename(columns={'Answer.h11': 'pAge'})




    df.loc[(df['Answer.h12.Male'] == True),'pGender'] = 'Male'
    df.loc[(df['Answer.h12.Female'] == True), 'pGender'] = 'Female'
    df.loc[(df['Answer.h12.Other'] == True), 'pGender'] = 'Other'
    df.loc[(df['Answer.h12.Prefer not to say'] == True), 'pGender'] = 'NP'

    df.loc[(df['Answer.h13.Hispanic or Latino'] == True), 'pRace'] = 'Latino'
    df.loc[(df['Answer.h14.American Indian or Alaska Native'] == True), 'pRace'] = 'Native'
    df.loc[(df['Answer.h14.Native Hawaiian or Other Pacific Islander'] == True), 'pRace'] = 'Islander'
    df.loc[(df['Answer.h14.Asian'] == True), 'pRace'] = 'Asian'
    df.loc[(df['Answer.h14.Black or African American'] == True), 'pRace'] = 'Black'
    df.loc[(df['Answer.h14.Prefer not to say'] == True), 'pRace'] = 'Black'
    df.loc[(df['Answer.h14.White'] == True), 'pRace'] = 'White'

    # df.loc[(df['Answer.h13.Not Hispanic or Latino'] == True), 'pRace'] = 'Not hispanic'


    df2['pId'] = pId
    df2['pRace'] = pRace
    df2['pGender'] = pGender
    df2['pAge'] = pAge

    df2 = df.drop_duplicates(subset='pId')
    # gender_list = []
    # for id in ppt.tolist():
    #     df_g = df[(df['pId'] == id)]
    #     gender_list.append(df_g['pGender'].tolist()[0])
    #
    # females = gender_list.count('Female')
    # males = gender_list.count('Male')

    print(df2['pRace'].value_counts())

    print(df2['pGender'].value_counts())

    print(df2['pAge'].mean())
    print(df2['pAge'].std())


    return df2

#
# def print_gender_demographics(df):
#     ppt = df['pId'].unique()
#
#     gender_list = []
#     for id in ppt.tolist():
#         df_g = df[(df['pId'] == id)]
#         gender_list.append(df_g['pGender'].tolist()[0])
#
#     females = gender_list.count('Female')
#     males = gender_list.count('Male')
#     total = females + males
#
#     print("F: {:.2f}% / M: {:.2f}%".format(females , males ))
#     print("F: {:.2f}% / M: {:.2f}%".format(females *100/ total, males *100 / total))
#
# def print_race_demographics(df):
#     ppt = df['pId'].unique()
#
#     race_list = []
#     for id in ppt.tolist():
#         df_g = df[(df['pId'] == id)]
#         race_list.append(df_g['pRace'].tolist()[0])
#
#     whites = race_list.count('White')
#     blacks = race_list.count('Black')
#     latinos = race_list.count('Latino')
#     asians = race_list.count('Asian')
#     natives = race_list.count('Native')
#
#
#     print("White: {:.2f}% / Black/African American: {:.2f}% / Hispanic/Latino: {:.2f}% / Asian: {:.2f}%/ Native American: {:.2f}%".format(whites * 100 / len(race_list), blacks * 100 / len(race_list),latinos * 100 / len(race_list), asians * 100 / len(race_list),natives * 100 / len(race_list)))

df = print_demographics("results3.csv")
# print_race_demographics(df)
# print_gender_demographics(df)