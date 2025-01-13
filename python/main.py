# # This is a sample Python script.
#
# # Press ⌃R to execute it or replace it with your code.
# # Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import matplotlib.pyplot as plt
import pandas as pd


import pandas as pd

# Define the path to the CSV file (or any other file format you are using)
file_path = '../csv/userevaluationOrdered.csv'

# Read the DataFrame
df = pd.read_csv(file_path)


# Define the path to the output CSV file
outfile = 'shuffledMTurk.csv'

# Define column groups
groups = [
    ['extroversion_video1', 'pers_ext_1', 'extroversion_video2', 'pers_ext_2'],
    ['neuroticism_video1', 'pers_neu_1', 'neuroticism_video2', 'pers_neu_2'],
    ['agreeableness_video1', 'pers_ag_1', 'agreeableness_video2', 'pers_agree_2'],
    ['conscientiousness_video1', 'pers_cons_1', 'conscientiousness_video2', 'pers_cons_2'],
    ['openness_video1', 'pers_open_1', 'openness_video2', 'pers_open_2']
]

# Create an empty DataFrame to store the shuffled groups
shuffled_df = pd.DataFrame()

# Shuffle each group and append to the shuffled DataFrame
for group in groups:
    shuffled_group = df[group].sample(frac=1).reset_index(drop=True)
    shuffled_df = pd.concat([shuffled_df, shuffled_group], axis=1)

# Write the shuffled DataFrame to a new CSV file
shuffled_df.to_csv(outfile, index=False)

