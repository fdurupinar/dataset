
import os
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

# Sample DataFrame


import shutil

# # Read the CSV file
# df = pd.read_csv('final_pers.csv')
#
#
#
# columns = ['File', 'openness' , 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism']
# label_df = df[columns]
#
# df2 = label_df.drop_duplicates(subset='File')
# df2.to_csv("labels.csv", index=False)
#
df = pd.read_csv('../csv/labels.csv')
# Define the personality traits
trait_columns = ['extroversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

# Initialize an empty DataFrame for the final output
final_df = pd.DataFrame()
base_url = "https://www.cs.umb.edu/~srikark/selected_videos/"
all_videos_selected = set()



# Define the personality traits
trait_columns = ['extroversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

##Select unique videos and write into a file
# get videos
def writeUnique(df_mturk):


    test_columns = ['extroversion_video1',	'extroversion_video2','neuroticism_video1',	'neuroticism_video2', 'agreeableness_video1',	'agreeableness_video2',	'conscientiousness_video1',	'conscientiousness_video2','openness_video1',	'openness_video2']

    # Combine the specified columns into a single series
    combined_series = pd.concat([df_mturk[col] for col in test_columns])

    # Drop duplicates
    unique_values = combined_series.drop_duplicates().reset_index(drop=True)

    # Write the result to a new CSV file
    unique_values.to_csv('unique_videos.csv', index=False, header=['video'])

def selectMTurkVideos(df, out_file):
    final_df = pd.DataFrame()

    for trait in trait_columns:
        left_videos = []
        right_videos = []

        left_values = []
        right_values = []


        for _ in range(80):

            #randomly select a label

            while True:
                labels = [0, 0.25, 0.5, 0.75, 1]
                lb1 =  np.random.choice(labels)
                labels2 = np.delete(labels, np.where(labels == lb1))
                lb2 = np.random.choice(labels2)

                df_l = df[df[trait] == lb1]
                df_r = df[df[trait] == lb2]

                if not df_l.empty and not df_r.empty:
                    v_l = df_l.sample(n=1).iloc[0]
                    v_r = df_r.sample(n=1).iloc[0]
                    break



            left_videos.append(base_url + v_l['File'])
            right_videos.append(base_url + v_r['File'])
            left_values.append(v_l[trait])
            right_values.append(v_r[trait])

            # Add selected videos to the set
            all_videos_selected.add(v_l['File'])
            all_videos_selected.add(v_r['File'])



        # Create a DataFrame for the current trait
        df_trait = pd.DataFrame({
            f"{trait}_video1": left_videos,
            f"{trait}_video2": right_videos,

            f"{trait}_value1": left_values,
            f"{trait}_value2": right_values,
        })

        # Append to the final DataFrame
        final_df = pd.concat([final_df, df_trait], axis=1)
    #
    # Save the final DataFrame to a CSV file
    final_df.to_csv(out_file, index=False)



########## MTurk Study 1 #####################################
df1 = pd.read_csv('../csv/labels.csv')
# selectMTurkVideos(df1, 'mturk_pers_test.csv')
# df_mturk = pd.read_csv('mturk_pers_test.csv')
# writeUnique(df_mturk)
##############################################################

########## MTurk Study 2 #####################################
# df2 = pd.read_csv('binned_regression_results.csv')
# selectMTurkVideos(df2, 'mturk2_input.csv')
df_mturk2 = pd.read_csv('../mturk2/mturk2_input_fixed.csv')
writeUnique(df_mturk2)
##############################################################


# # Define the base directory where the videos are stored
# base_directory = '/Users/srikarkodavati/Desktop/face_dataset/user_output'
#
# # Define the new directory where the videos will be copied
# new_directory = '/Users/srikarkodavati/Desktop/face_dataset/selected_videos'
#
# # Create the new directory if it doesn't exist
# if not os.path.exists(new_directory):
#     os.makedirs(new_directory)
#
# # Function to find the full path of a video file
# def find_video_path(filename, base_directory):
#     for root, dirs, files in os.walk(base_directory):
#         if filename in files:
#             return os.path.join(root, filename)
#     return None
#
# count = 0
#
# # Copy the selected videos to the new directory
# for video in all_videos_selected:
#     video_path = find_video_path(video, base_directory)
#     if video_path:
#         shutil.copy(video_path, new_directory)
#         count += 1
#         print(f"Copied {count} files so far: {video}")
#     else:
#         print(f"Video file {video} not found in the base directory.")
#
#
