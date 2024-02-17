# Taylor Swift Song Album Classifier

![Image Description](https://s2-valor.glbimg.com/Yua9P2Y1h5bN9SS8x_KxUMdxTIs=/0x0:3000x2000/888x0/smart/filters:strip_icc()/i.s3.glbimg.com/v1/AUTH_63b422c2caee4269b8b34177e8876b93/internal_photos/bs/2023/O/B/pBmpraTpusPFuQzb5gqw/399856954.jpg)

This project is a very simple machine learning model that classifies Taylor Swift songs into their respective albums based on their lyrics. The ideia was not to reinvent the wheel, was simply to to create a fun and educational project for fans of Taylor Swift and also learn some basic of the NLP and Machine Learning.

## How It Works

1. **Data Collection**: The lyrics data for Taylor Swift songs is collected from various sources.
2. **Data Preprocessing**: The lyrics are cleaned by converting them to lowercase, removing punctuation, and tokenizing them into words.
3. **Word Embeddings**: Word2Vec is used to create word embeddings for each song's lyrics, which are then aggregated to create a single vector representation for each song.
4. **Model Training**: The SVM (Support Vector Machine) model is trained using the song vectors and their corresponding album labels. (In the future I'll be adding KNN to test and compare)
5. **Model Evaluation**: The model is evaluated using accuracy, precision, recall, and F1-score metrics to measure its performance.
6. **Prediction**: Finally, the model is used to predict the album for each song in a test set.

## Setup Instructions

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run `python svm.py` to preprocess the data, train the model, and make predictions.

## Project Structure

- `svm.py`: The main Python script that contains the data preprocessing, model training, and evaluation code.
- `taylor_swift_lyrics.xlsx`: The dataset containing Taylor Swift song lyrics and their corresponding album labels.
- `requirements.txt`: A list of Python packages required to run the project.

## Results

I'm not able to take the results from the model yet cause I'm still working on it, even though is pretty simple.
<!-- The model achieves an accuracy of X%, precision of Y%, recall of Z%, and an F1-score of W%. -->

## Future Improvements

- Experiment with different word embedding techniques.
- Try different machine learning models.
- Include additional features such as song length or sentiment analysis of lyrics.

## Contributions

Contributions are welcome! If you have any ideas for improvements or new features, feel free to submit a pull request.

## Credits

This project was created by Marco Tulio. Special thanks to Vinicius Schiavon for help and to @adashofdata who provided the dataset. 

That's all for now folks! If you have any questions or suggestions, feel free to reach out to me. Thanks for reading! 🎵🎶

![ImageDescription](https://i.pinimg.com/originals/3a/f7/93/3af79303f82c777ae7ebac1b2d9fa763.jpg)