import pandas as pd
import numpy as np
import string
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


def songs_pre_precessed():
    # Load the data, make it all lowercase, and remove punctuation
    df = pd.read_excel('/home/marcotuiio/TaylorSwift_Guesser/taylor_swift_lyrics.xlsx')

    df['Lyrics'] = df['Lyrics'].str.lower()
    df['Lyrics'] = df['Lyrics'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))
    stop_words = set(stopwords.words('english'))
    # stop_words.add(["oh", "ah"])
    stop_words.add("oh")
    stop_words.add("ah")
    stop_words.add("yeah")
    stop_words.add("hey")
    
    df['Lyrics'] = df['Lyrics'].apply(lambda x: ''.join([char for char in x if char not in stop_words]))

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
    df.to_excel('taylor_swift_lyrics_preprocessed.xlsx', index=False)

    return df

def process_all_songs():
    df = pd.read_csv('/home/marcotuiio/TaylorSwift_Guesser/combined_dataset.csv')

    df['Lyric'] = df['Lyric'].str.lower()
    df['Lyric'] = df['Lyric'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    stop_words = set(stopwords.words('english'))
    stop_words.update(["oh", "ah", "yeah", "hey"])  # Add common words to stop_words

    df['Lyric'] = df['Lyric'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

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
    df.to_excel('taylor_swift_lyrics_preprocessed.xlsx', index=False)

    return df
