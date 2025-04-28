import pandas as pd

def compare_csv(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Sort the dataframes to ensure rows are in the same order
    df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
    df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

    # Compare the two dataframes
    comparison = df1.eq(df2)

    # Calculate similarity percentage
    similarity = comparison.sum().sum() / (comparison.shape[0] * comparison.shape[1]) * 100

    print(f'Similarity: {similarity}%')

# Call the function with your file paths
compare_csv('predictions_adam.csv', 'predictions_eva.csv')