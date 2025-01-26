
import os
import pandas as pd
from statsmodels.stats.multitest import multipletests

from scipy.stats import chi2_contingency


#
# dfResponses1 = pd.read_csv('validResults2Part1.csv')
# dfResponses2 = pd.read_csv('validResults2Part2.csv')
# dfResponses3 = pd.read_csv('validResults2Part3.csv')


# dfResponses = pd.concat([dfResponses1, dfResponses2, dfResponses3])

# dfResponses = pd.read_csv('resultsSrikar.csv')

# ############################MIG RESULTS ##################################
dfResponses = pd.read_csv('../mturk1/validResults3.csv') #MIG results
dfLabels = pd.read_csv('../csv/labels.csv') # Actual results
##############################################################################
# dfResponses = pd.read_csv('../mturk3/mturk3validResults.csv') #MIG results
# dfLabels = pd.read_csv('../csv/labels.csv') # Actual results


#
# #Replace
# dfResponses = dfResponses.rename(columns={'Answer.hO.left': 'openness.left', 'Answer.hO.right': 'openness.right',  'Answer.hO.equal': 'openness.equal',
# 'Answer.hC.left': 'conscientiousness.left', 'Answer.hC.right': 'conscientiousness.right',  'Answer.hC.equal': 'conscientiousness.equal',
# 'Answer.hE.left': 'extroversion.left', 'Answer.hE.right': 'extroversion.right',  'Answer.hE.equal': 'extroversion.equal',
# 'Answer.hA.left': 'agreeableness.left', 'Answer.hA.right': 'agreeableness.right','Answer.hA.equal': 'agreeableness.equal',
# 'Answer.hN.left': 'neuroticism.left', 'Answer.hN.right': 'neuroticism.right', 'Answer.hN.equal': 'neuroticism.equal'})

dfResponses = dfResponses.rename(columns={'Answer.hO.left': 'openness.left', 'Answer.hO.right': 'openness.right',
'Answer.hC.left': 'conscientiousness.left', 'Answer.hC.right': 'conscientiousness.right',
'Answer.hE.left': 'extroversion.left', 'Answer.hE.right': 'extroversion.right',
'Answer.hA.left': 'agreeableness.left', 'Answer.hA.right': 'agreeableness.right',
'Answer.hN.left': 'neuroticism.left', 'Answer.hN.right': 'neuroticism.right',})


answer_columns = [
    'openness.left', 'openness.right', 'conscientiousness.left', 'conscientiousness.right',
    'extroversion.left', 'extroversion.right', 'agreeableness.left', 'agreeableness.right',
    'neuroticism.left', 'neuroticism.right'
]

value_counts = {
    'openness': {'left': 0, 'right': 0},
    'conscientiousness': {'left': 0, 'right': 0},
    'extroversion': {'left': 0, 'right': 0},
    'agreeableness': {'left': 0, 'right': 0},
    'neuroticism': {'left': 0, 'right': 0}
}


# Function to get the trait value from dfLabels
def get_trait_value(file, trait):
    value = dfLabels[dfLabels['File'] == file][trait]
    return value.values[0] if not value.empty else None


# Compare the traits and count where the left file's trait value is greater than the right file's trait value
for index, row in dfResponses.iterrows():
    for trait in value_counts.keys():
        left_file = row[f'Input.{trait}_video1'].rsplit('/', 1)[-1]
        right_file = row[f'Input.{trait}_video2'].rsplit('/', 1)[-1]

        left_trait_value = get_trait_value(left_file, trait)
        right_trait_value = get_trait_value(right_file, trait)

        # if row[trait+'.equal'] == False:
        if left_trait_value is not None and right_trait_value is not None:
            if left_trait_value > right_trait_value:
                value_counts[trait]['left'] += 1
            elif right_trait_value > left_trait_value:
                value_counts[trait]['right'] += 1



# # Display the counts
for trait, count in value_counts.items():
    print(f"{trait.capitalize()} - Actual Left > Right count: {count}")


# Initialize a dictionary to store the counts of left and right responses
left_right_counts = {}
actual_left_right_counts = {}

# Count left and right responses
for col in answer_columns:
    trait = col.split('.')[0]

    countLeft = (dfResponses[trait+'.left'] == True).sum()
    countRight = (dfResponses[trait+'.right'] == True).sum()

    left_right_counts[trait] = {'left': countLeft,'right': countRight}

#
# # Display the counts of left and right responses
for trait, counts in left_right_counts.items():
    print(f"{trait} Predicted - Left: {counts['left']}, Predicted- Right: {counts['right']}")




# Define the traits and their corresponding columns for value comparisons
traits = [  'openness', 'conscientiousness','extroversion','agreeableness',  'neuroticism']


# Combine the counts into a contingency table
contingency_table = pd.DataFrame(index=traits, columns=['Left', 'Right', 'Actual Left', 'Actual Right'])

for trait in traits:
    contingency_table.at[trait, 'Left'] = left_right_counts[trait]['left']
    contingency_table.at[trait, 'Right'] = left_right_counts[trait]['right']
    contingency_table.at[trait, 'Actual Left'] = value_counts[trait]['left']
    contingency_table.at[trait, 'Actual Right'] = value_counts[trait]['right']


# Calculate the sums for each column
sums = contingency_table.sum()

# Append the sums as a new row 'all'
contingency_table.loc['all'] = sums
# Display the contingency table
print("\nContingency Table:")
print(contingency_table)



# Create a dictionary to hold individual 2x2 tables
individual_tables = {}

# Iterate over the rows of the contingency table
for trait in contingency_table.index:
    # Extract the row for the current trait
    row = contingency_table.loc[trait]
    # Create a 2x2 DataFrame
    table = pd.DataFrame({
        'Predicted Left/Right': ['Left', 'Right'],
        'Actual Left/Right': [row['Left'], row['Right']],
        'Actual': [row['Actual Left'], row['Actual Right']]
    })
    # Set the index to "Predicted Left/Right"
    table.set_index('Predicted Left/Right', inplace=True)
    # Store the table in the dictionary
    individual_tables[trait] = table

# List to collect p-values
pvals = []

# Compute chi-squared test p-values and collect them
for trait, table in individual_tables.items():
    print(f"\nTable for {trait.capitalize()}:")
    print(table)

    # Convert the table to the appropriate format for chi2_contingency
    contingency = table.values.reshape(2, 2)
    stat, p, dof, expected = chi2_contingency(contingency)

    pvals.append(p)
    print("p value for " + trait + " is " + str(p))

# Apply FDR correction
_, corrected_pvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

# Update the individual tables with the corrected p-values
for i, (trait, table) in enumerate(individual_tables.items()):
    table['p-value'] = [pvals[i]] * len(table)
    table['corrected p-value'] = [corrected_pvals[i]] * len(table)

# Display the updated tables with corrected p-values
for trait, table in individual_tables.items():
    print(f"\nUpdated table for {trait.capitalize()}:")
    print(table)


# Apply FDR correction
_, corrected_p_values, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
multipletests(pvals, alpha=0.05, method='fdr_bh')


def getValidData(fileIn, fileOut):
    # df1 = pd.read_csv('results1.csv')
    # df2 = pd.read_csv('results2.csv')
    input_columns = ['Input.extroversion_video1', 'Input.extroversion_video2','Input.neuroticism_video1', 'Input.neuroticism_video2', 'Input.agreeableness_video1', 'Input.agreeableness_video2', 'Input.conscientiousness_video1', 'Input.conscientiousness_video2', 'Input.openness_video1', 'Input.openness_video2' ]
    # answer_columns = ['WorkerId', 'Answer.hO.equal','Answer.hO.left', 'Answer.hO.right' , 'Answer.hC.equal','Answer.hC.left', 'Answer.hC.right',  'Answer.hE.equal','Answer.hE.left', 'Answer.hE.right', 'Answer.hA.equal','Answer.hA.left', 'Answer.hO.right',  'Answer.hO.equal','Answer.hO.left', 'Answer.hA.right', 'Answer.hN.equal','Answer.hN.left', 'Answer.hN.right', 'Answer.hT.equal' ]
    answer_columns = ['WorkerId', 'Answer.hO.left', 'Answer.hO.right' , 'Answer.hC.left', 'Answer.hC.right',  'Answer.hE.left', 'Answer.hE.right', 'Answer.hA.left', 'Answer.hO.right',  'Answer.hO.left', 'Answer.hA.right','Answer.hN.left', 'Answer.hN.right', 'Answer.hTLeft.left', 'Answer.hTRight.right' ]

    # df = pd.concat([df1, df2], axis=0)
    df = pd.read_csv(fileIn)


    # eliminate rejected data

    # Reset index to ensure unique indices
    df = df.reset_index(drop=True)


    # Filter the DataFrame where 'Answer.hT.equal' is True
    # Identify workerIds with False in Answer.hT.equal
    rejected_worker_ids = df[(df['Answer.hTLeft.left'] == False) | (df['Answer.hTRight.right'] == False) ]['WorkerId'].unique()


    # Drop rows for these workerIds
    filtered_df = df[~df['WorkerId'].isin(rejected_worker_ids)]


    # Select the specified columns
    filtered_df = filtered_df[input_columns + answer_columns]

    # Write the filtered DataFrame to a new CSV file
    filtered_df.to_csv(fileOut, index=False)

    print(rejected_worker_ids)



# getValidData('results3.csv', 'validResults3.csv')# Results used in MIG


# getValidData('../mturk2/mturk2_results.csv', '../mturk2/mturk2ValidResults.csv')

getValidData('../mturk3/mturk3_results.csv', '../mturk3/mturk3ValidResults.csv')

# getValidData('results2Part1.csv', 'validResults2Part1.csv')
# getValidData('results2Part2.csv', 'validResults2Part2.csv')
# getValidData('results2Part3.csv', 'validResults2Part3.csv')
#
# getValidData('results1Part1.csv', 'validResults1Part1.csv')
# getValidData('results1Part2.csv', 'validResults1Part2.csv')
