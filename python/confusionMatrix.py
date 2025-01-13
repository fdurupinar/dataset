
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


def confusionMatrix(fileIn):
    df = pd.read_csv(fileIn)

    # List of traits
    traits = ['o', 'c', 'e', 'a', 'n']

    # Create confusion matrices for each trait
    for trait in traits:
        y_true = df[f'{trait}.true']
        y_pred = df[f'{trait}.answer']
        cm = confusion_matrix(y_true, y_pred, labels=['left', 'right'])

        # Create a DataFrame for the confusion matrix
        cm_df = pd.DataFrame(cm, index=['True: left', 'True: right'],
                             columns=['Pred: left', 'Pred: right'])

        # Display the confusion matrix
        print(f"\nConfusion Matrix for {trait.upper()}:")
        print(cm_df)

        # Plot the confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {trait.upper()}')
        plt.show()



def precisionAccuracy(fileIn):
    # df1 = pd.read_csv('formattedResults2.csv')
    # df2 = pd.read_csv('formattedResults3.csv')
    #
    # df = pd.concat([df1, df2])
    df = pd.read_csv(fileIn)

    # df = pd.read_csv('formattedResults3.csv')
    # Define the traits
    traits = ['o', 'c', 'e', 'a', 'n']


    results = {}
    pvals = []
    # Calculate confusion matrix, accuracy, and precision for each trait
    for trait in traits:
        # Filter out rows where the answer is 'equal'
        # filtered_df = df[df[f'{trait}.answer'] != 'equal']
        filtered_df = df
        y_true = filtered_df[f'{trait}.true']
        y_pred = filtered_df[f'{trait}.answer']
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Calculate precision (handle cases with precision zero-division)
        precision = precision_score(y_true, y_pred, pos_label='left', average='binary', zero_division=1)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['left', 'right'])
        contingency_table = pd.DataFrame(cm, index=['Actual Left', 'Actual Right'],
                                         columns=['Predicted Left', 'Predicted Right'])

        # Calculate entropy


        # Calculate p-value for accuracy using binomial test
        n = len(y_true)
        p_value_accuracy = binom_test(sum(y_true == y_pred), n, p=0.5, alternative='greater')

        # Convert the table to the appropriate format for chi2_contingency

        stat, p, dof, expected = chi2_contingency(contingency_table)


        pvals.append(p)
        results[trait] = {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'p_value_accuracy': p_value_accuracy,
            'p_value_chi':p  # Placeholder for precision p-value

        }
    _, corrected_pvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    print(corrected_pvals)
    # Display the results
    for trait, metrics in results.items():
        print(f"\nMetrics for {trait.capitalize()}:")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"P-value for Accuracy: {metrics['p_value_accuracy']:.4f}")
        print(f"P-value for Chi2: {metrics['p_value_chi']}")  # Placeholder



    print(contingency_table)

def reformatFiles(fileIn, fileOut, dfLabels):

    dfResponses = pd.read_csv(fileIn)

    #rename for easy editing
    dfResponses = dfResponses.rename(

        columns={'Answer.hO.left': 'o.left', 'Answer.hO.right': 'o.right',
                 'Answer.hC.left': 'c.left', 'Answer.hC.right': 'c.right',
                 'Answer.hE.left': 'e.left', 'Answer.hE.right': 'e.right',
                 'Answer.hA.left': 'a.left', 'Answer.hA.right': 'a.right',
                 'Answer.hN.left': 'n.left', 'Answer.hN.right': 'n.right',
                 'Input.openness_video1': 'o.video.left', 'Input.openness_video2': 'o.video.right',
                 'Input.conscientiousness_video1': 'c.video.left', 'Input.conscientiousness_video2': 'c.video.right',
                 'Input.extroversion_video1': 'e.video.left', 'Input.extroversion_video2': 'e.video.right',
                 'Input.agreeableness_video1': 'a.video.left', 'Input.agreeableness_video2': 'a.video.right',
                 'Input.neuroticism_video1': 'n.video.left', 'Input.neuroticism_video2': 'n.video.right'
                 })

    dfLabels = dfLabels.rename(columns={'openness':'o', 'conscientiousness':'c', 'extroversion':'e', 'agreeableness':'a', 'neuroticism':'n'})


    df = pd.DataFrame( columns=['workerId', 'video.left', 'video.right', 'o.true', 'c.true', 'e.true', 'a.true', 'n.true', 'o.answer', 'c.answer', 'e.answer', 'a.answer', 'n.answer' ])

    traits = ['o', 'c','e','a','n']

    # Function to get the trait value from dfLabels
    def get_trait_value(file, trait):
        value = dfLabels[dfLabels['File'] == file][trait]
        return value.values[0] if not value.empty else None

    # List to collect new rows
    new_rows = []
    # Compare the traits and count where the left file's trait value is greater than the right file's trait value
    for index, row in dfResponses.iterrows():
        new_row = {'workerId': row['WorkerId']}
        for trait in traits:
            left_file = row[f'{trait}.video.left'].rsplit('/', 1)[-1]
            right_file = row[f'{trait}.video.right'].rsplit('/', 1)[-1]

            left_trait_value = get_trait_value(left_file, trait)
            right_trait_value = get_trait_value(right_file, trait)

            # if row[trait+'.equal'] == False:
            new_row['video.left'] = left_file
            new_row['video.right'] = right_file

            if row[trait + '.left'] == True:
                new_row[trait + '.answer'] = 'left'
            elif row[trait + '.right'] == True:
                new_row[trait + '.answer'] = 'right'
            else:
                new_row[trait + '.answer'] = 'equal'

            if left_trait_value is not None and right_trait_value is not None:
                if left_trait_value > right_trait_value:
                    new_row[trait+'.true'] = 'left'
                elif right_trait_value > left_trait_value:
                    new_row[trait + '.true'] = 'right'
                else: # this will never happen
                    new_row[trait + '.true'] = 'equal'

        new_rows.append(new_row)

    # Convert the list of new rows to a DataFrame
    new_rows_df = pd.DataFrame(new_rows)

    # Concatenate the new rows DataFrame with the original DataFrame
    df = pd.concat([df, new_rows_df], ignore_index=True)


    df.to_csv(fileOut, index=False)



def calculatePrecision(diff, dfResponses, dfLabels):

    #rename for easy editing
    dfResponses = dfResponses.rename(columns={'Answer.hO.left': 'o.left', 'Answer.hO.right': 'o.right',  'Answer.hO.equal': 'o.equal',
    'Answer.hC.left': 'c.left', 'Answer.hC.right': 'c.right',  'Answer.hC.equal': 'c.equal',
    'Answer.hE.left': 'e.left', 'Answer.hE.right': 'e.right',  'Answer.hE.equal': 'e.equal',
    'Answer.hA.left': 'a.left', 'Answer.hA.right': 'a.right','Answer.hA.equal': 'a.equal',
    'Answer.hN.left': 'n.left', 'Answer.hN.right': 'n.right', 'Answer.hN.equal': 'n.equal',
    'Input.openness_video1':'o.video.left', 'Input.openness_video2':'o.video.right',
    'Input.conscientiousness_video1':'c.video.left', 'Input.conscientiousness_video2':'c.video.right',
    'Input.extroversion_video1':'e.video.left', 'Input.extroversion_video2':'e.video.right',
    'Input.agreeableness_video1':'a.video.left', 'Input.agreeableness_video2':'a.video.right',
    'Input.neuroticism_video1':'n.video.left', 'Input.neuroticism_video2':'n.video.right'
    })

    dfLabels = dfLabels.rename(columns={'openness':'o', 'conscientiousness':'c', 'extroversion':'e', 'agreeableness':'a', 'neuroticism':'n'})

    df = pd.DataFrame(columns = ['video', 'trait', 'selectedCnt', 'totalCnt'])
    # df = pd.DataFrame( columns=['workerId', 'o.video.left', 'o.video.right', 'o.true', 'c.true', 'e.true', 'a.true', 'n.true', 'o.answer', 'c.answer', 'e.answer', 'a.answer', 'n.answer' ])

    traits = ['o', 'c','e','a','n']

    # Function to get the trait value from dfLabels
    def get_trait_value(file, trait):
        value = dfLabels[dfLabels['File'] == file][trait]
        return value.values[0] if not value.empty else None

    # List to collect new rows

    correctCnt = {'o':0, 'c':0, 'e':0, 'a':0, 'n':0}
    incorrectCnt = {'o': 0, 'c': 0, 'e': 0, 'a': 0, 'n': 0}
    totalCnt = {'o': 0, 'c': 0, 'e': 0, 'a': 0, 'n': 0}
    accuracy = {'o': 0, 'c': 0, 'e': 0, 'a': 0, 'n': 0}
    # Compare the traits and count where the left file's trait value is greater than the right file's trait value
    for index, row in dfResponses.iterrows():

        for trait in traits:
            left_file = row[f'{trait}.video.left'].rsplit('/', 1)[-1]
            right_file = row[f'{trait}.video.right'].rsplit('/', 1)[-1]

            if not df[(df['video'] == left_file) & (df['trait'] == trait)].empty:
                df.loc[(df['video'] == left_file) & (df['trait'] == trait), 'totalCnt'] += 1

            else:
                new_row = {'video': left_file, 'trait': trait, 'selectedCnt':0, 'totalCnt':1}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            if not df[(df['video'] == right_file) & (df['trait'] == trait)].empty:
                df.loc[(df['video'] == right_file) & (df['trait'] == trait), 'totalCnt'] += 1
            else:
                new_row = {'video': right_file, 'trait': trait, 'selectedCnt':0, 'totalCnt': 1}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            if row[trait + '.left'] == True:
                df.loc[(df['video'] == left_file) & (df['trait'] == trait), 'selectedCnt'] += 1

            if row[trait + '.right'] == True:
                df.loc[(df['video'] == right_file) & (df['trait'] == trait), 'selectedCnt'] += 1

            left_trait_value = get_trait_value(left_file, trait) # labeled trait value
            right_trait_value = get_trait_value(right_file, trait) # labeled trait value


            if left_trait_value is not None and right_trait_value is not None:
                if abs(left_trait_value - right_trait_value) > diff: # only consider the cases where differences are bigger than dif
                        if left_trait_value > right_trait_value:
                            if row[trait + '.left'] == True:
                                correctCnt[trait] += 1
                            elif row[trait + '.right'] == True:
                                incorrectCnt[trait] += 1
                        elif left_trait_value < right_trait_value:
                            if row[trait + '.right'] == True:
                                correctCnt[trait] += 1
                            elif row[trait + '.left'] == True:
                                incorrectCnt[trait] += 1

                # else: # if they are equal
                #     if row[trait + '.equal'] == True:
                #         correctCnt[trait] += 1
                #     else:
                #         incorrectCnt[trait] += 1



    for trait in traits:
        totalCnt[trait] = correctCnt[trait] + incorrectCnt[trait]
        if totalCnt[trait] > 0 :
            accuracy[trait] = correctCnt[trait] /totalCnt[trait]
        else:
            accuracy[trait] = None



    print(correctCnt)
    print(incorrectCnt)
    print(totalCnt)
    print(accuracy)
    return df



########################## Results for Study 1 - Classification #################################
# dfLabels = pd.read_csv('labels.csv')
# dfResponses = pd.read_csv('validResults3.csv')
# reformatFiles("validResults2.csv", "formattedResults2.csv", dfLabels)
# reformatFiles("validResults3.csv", "formattedResults3.csv", dfLabels)
# calculatePrecision(0,dfResponses, dfLabels )
# confusionMatrix('formattedResults2.csv')
# confusionMatrix('formattedResults3.csv')
#################################################################################################

########################## Results for Study 2 - Regression #################################

dfLabels = pd.read_csv('../mturk2/binned_regression_results.csv')
dfResponses = pd.read_csv('../mturk2/mturk2ValidResults.csv')
dfFormatted = calculatePrecision(0,dfResponses, dfLabels)
dfFormatted.to_csv("../mturk2/mturk2formattedResults.csv", index=False)
# dfResponses = confusionMatrix('../mturk2/mturk2formattedResults.csv')


#################################################################################################






# calculatePrecision(0.5,'formattedResults3.csv')

# calculatePrecision(0.5,'formattedResults3.csv' )
# precisionAccuracy('formattedResults2.csv')
# # calculatePrecision('fileIn')
# #
