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

    for _, row in dfResponses.iterrows():
        for trait in traits:
            left_file = row[f'{trait}.video.left'].rsplit('/', 1)[-1]
            right_file = row[f'{trait}.video.right'].rsplit('/', 1)[-1]


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


    return df



def getNewLabels(fileIn, fileOut, actualLabels):
    """
    :param fileIn:
    :param fileOut:
    :param actualLabels: Needed because we want the existing values unchanged in the user study
    :return:
    """
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

#################### MTurk 2 ########################################
# dfResponses = pd.read_csv('../mturk2/mturk2ValidResults.csv')
# dfFormatted = calculateSelectionCountPerVideo(dfResponses)
# dfFormatted.to_csv("../mturk2/mturk2formattedResults.csv", index=False)
# getNewLabels("../mturk2/mturk2formattedResults.csv", "../mturk2/mturk2UserLabels.csv", '../mturk2/binned_regression_results.csv')

#################### MTurk 3 ########################################
dfResponses = pd.read_csv('../mturk3/mturk3ValidResults.csv')
dfFormatted = calculateSelectionCountPerVideo(dfResponses)
dfFormatted.to_csv("../mturk3/mturk3formattedResults.csv", index=False)
getNewLabels("../mturk3/mturk3formattedResults.csv", "../mturk3/mturk3UserLabels.csv", '../mturk3/binned_regression_results_3.csv')

