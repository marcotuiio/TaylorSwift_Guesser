import os
import pandas as pd

# Path to the directory containing the CSV files
directory_path = '/home/marcotuiio/TaylorSwift_Guesser/archive/csv'

# Initialize an empty list to store all DataFrames
dfs = []

# Iterate over all files in the directory
for file in os.listdir(directory_path):
    if file.endswith('.csv'):
        print(f'Processing file: {file}')
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)

        # Assuming the CSV files have columns 'artist', 'song_name', 'lyrics'
        # Add a new column 'label' with 'yes' if the artist is 'Taylor Swift', 'no' otherwise
        df['label'] = 'yes' if 'Taylor Swift' in df['Artist'].values else 'no'

        # Keep only the required columns
        df = df[['Artist', 'Title', 'Lyric', 'label']]

        dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
result_df = pd.concat(dfs, ignore_index=True)

# Reset index
result_df.reset_index(drop=True, inplace=True)

# Save the result DataFrame to a CSV file
result_df.to_csv('combined_dataset.csv', index=False)
