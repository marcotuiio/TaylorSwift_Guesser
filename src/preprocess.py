import pandas as pd
import numpy as np
import string
import re
import os
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def word_to_vector(word, model):
    try:
        return model.wv[word]
    except KeyError:
        return None


def aggregate_vectors(tokens, model):
    vectors = [word_to_vector(token, model) for token in tokens]
    vectors = [vec for vec in vectors if vec is not None]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def ts_songs_pre_precessed():
    # Load the data, make it all lowercase, and remove punctuation
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    file_to_read = os.path.join(project_root, 'assets', 'taylor_swift_lyrics.xlsx')
    df = pd.read_excel(file_to_read)

    df['Lyrics'] = df['Lyrics'].str.lower()
    df['Lyrics'] = df['Lyrics'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))

    # Tokenize the lyrics before removing stopwords
    df['tokens'] = df['Lyrics'].apply(word_tokenize)

    stop_words = set(stopwords.words('english'))
    stop_words.update(["oh", "ah", "yeah", "hey"])  # Add common words to stop_words

    df['Lyrics'] = df['tokens'].apply(lambda x: ' '.join([word for word in x if word.lower() not in stop_words]))

    # print(df)

    # Tokenize the lyrics, I'll be using word2vec to create word embeddings
    df['tokens'] = df['Lyrics'].apply(word_tokenize)

    # Train Word2Vec model
    # vector_size: The dimensionality of the word vectors
    # window: The maximum distance between the current and predicted word within a sentence
    # min_count: Ignores all words with total frequency lower than this
    # sg: Training algorithm: 1 for skip-gram; 0 for CBOW
    model = Word2Vec(df['tokens'], vector_size=200, window=150, min_count=1, sg=0)

    # Apply aggregation function to the 'tokens' column
    df['song_vector'] = df['tokens'].apply(lambda x: aggregate_vectors(x, model))
    df['Album'] = df['Album'].astype(str)

    file_to_write = os.path.join(project_root, 'assets', 'taylor_swift_lyrics_preprocessed.xlsx')
    df.to_excel(file_to_write, index=False)

    return df

def generate_combined_dataset():
    # Path to the directory containing the CSV files
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    directory_path = os.path.join(project_root, 'assets', 'archive')

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
            
            ### This is to filter bad data, but it may still need adjustments
            
            df = df[df['Lyric'].str.contains('lyrics for this song have yet to be released please check back once the song has been released') == False]
            df = df[df['Lyric'].str.contains('unreleased') == False]
            df = df[df['Lyric'].str.contains('instrumental') == False]
            df = df[df['Lyric'].str.contains('background vocal') == False]
            df = df[df['Lyric'].str.contains('lyrics from snippet') == False]  
            df = df[df['Lyric'].str.contains('not yet available') == False]
            df = df[df['Lyric'].str.contains('not yet released') == False]
            df = df[df['Lyric'].str.contains('not confirmed') == False]
            df = df[df['Lyric'].str.contains('rihanna') == False]
            df = df[df['Lyric'].str.contains('tba') == False]
            df = df[df['Lyric'].str.contains('to be announced') == False]
            df = df[df['Lyric'].str.contains('soon') == False]
            df = df[df['Lyric'].str.contains('beyoncé') == False]

            # Remove rows where 'Title' contains '*'
            df = df[df['Title'].str.contains('\*', regex=True, na=False) == False]

            df = df[df['Title'].str.contains('Legendary') == False]
            df = df[df['Title'].str.contains('Rape') == False]
            df = df[df['Title'].str.contains('Eminem') == False]
            df = df[df['Title'].str.contains('Premonition') == False]
            df = df[df['Title'].str.contains('Chromatica') == False]
            # Assuming df is your DataFrame with a 'Lyric' column containing the lyrics

            df = df[df['Artist'].str.contains('BTS') == False]

            df['Lyric'] = df['Lyric'].str.replace('Beyoncé', 'Beyonce')
            df['Artist'] = df['Artist'].str.replace('Beyoncé', 'Beyonce')

            # Keep only the required columns
            df = df[['Artist', 'Title', 'Lyric', 'label']]

            dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    result_df = pd.concat(dfs, ignore_index=True)

    # Reset index
    result_df.reset_index(drop=True, inplace=True)

    # Save the result DataFrame to a CSV file
    file_to_write = os.path.join(project_root, 'assets', 'combined_dataset.xlsx')
    result_df.to_excel(file_to_write, index=False)


def process_all_songs():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    if (os.path.exists(os.path.join(project_root, 'assets', 'combined_dataset.xlsx')) == False):
        generate_combined_dataset()
        
    file_to_read = os.path.join(project_root, 'assets', 'combined_dataset.xlsx')
    df = pd.read_excel(file_to_read)

    df['Lyric'] = df['Lyric'].str.lower()
    df['Lyric'] = df['Lyric'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))

    # Tokenize the lyrics before removing stopwords
    df['tokens'] = df['Lyric'].apply(word_tokenize)

    stop_words = set(stopwords.words('english'))
    stop_words.update(["oh", "ah", "yeah", "hey"])  # Add common words to stop_words

    df['Lyric'] = df['tokens'].apply(lambda x: ' '.join([word for word in x if word.lower() not in stop_words]))

    # print(df)

    # Tokenize the lyrics, I'll be using word2vec to create word embeddings
    df['tokens'] = df['Lyric'].apply(word_tokenize)

    # Train Word2Vec model
    # vector_size: The dimensionality of the word vectors
    # window: The maximum distance between the current and predicted word within a sentence
    # min_count: Ignores all words with total frequency lower than this
    # sg: Training algorithm: 1 for skip-gram; 0 for CBOW
    model = Word2Vec(df['tokens'], vector_size=200, window=150, min_count=1, sg=0)

    # Apply aggregation function to the 'tokens' column
    df['song_vector'] = df['tokens'].apply(lambda x: aggregate_vectors(x, model))

    file_to_write = os.path.join(project_root, 'assets', 'combined_dataset_preprocessed.xlsx')
    df.to_excel(file_to_write, index=False)

    return df
