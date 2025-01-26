import pandas as pd
import pandas as pd

def revertRejections(fileIn, fileOut):
    # Read the input file
    df = pd.read_csv(fileIn)

    # Reset index to ensure unique indices
    df = df.reset_index(drop=True)

    # Identify rejected rows
    rejected_columns = df[(df['Answer.hTLeft.left'] == False) | (df['Answer.hTRight.right'] == False)]
    rejected_columns = rejected_columns.copy()
    rejected_columns['Reject'] = 'X'

    # Identify accepted rows (opposite condition)
    accepted_columns = df[(df['Answer.hTLeft.left'] == True) & (df['Answer.hTRight.right'] == True)]
    accepted_columns = accepted_columns.copy()
    accepted_columns['Approve'] = 'X'

    # Combine rejected and accepted rows
    updated_df = pd.concat([rejected_columns, accepted_columns], axis=0)

    # Write the updated DataFrame to the output file
    updated_df.to_csv(fileOut, index=False)
revertRejections('Batch_5289304_batch_results.csv', 'updated_Batch_5289304_batch_results.csv')