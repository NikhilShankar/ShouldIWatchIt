import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Prediction:
    def __init__(self, model, preprocess, metrics_folder, min_word_count=10):
        self.min_word_count = min_word_count
        self.model = model
        self.preprocess = preprocess
        self.metrics_folder = metrics_folder

    def predict(self, youtube_url):
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(youtube_url, sort_by=SORT_BY_POPULAR)
        comments_df = pd.DataFrame(comments)
        comments_df.rename(columns={'text': 'Comment'}, inplace=True)
        

        comments_df = self.preprocess.preprocess_dataframe(comments_df)

        X_comments = comments_df["Comment"]
        #Tokenizing for prediction purpose
        tokenizer = Tokenizer(num_words=10000)  # Set a max number of words
        tokenizer.fit_on_texts(X_comments)

        X_comments_train_seq = tokenizer.texts_to_sequences(X_comments)
        X_comments_train_padded = pad_sequences(X_comments_train_seq, padding='post', maxlen=100)  # Adjust maxlen as needed

        predictions = self.model.model.predict(X_comments_train_padded)

        sentiments = np.argmax(predictions[0], axis=1)
        outcomes = np.argmax(predictions[1], axis=1)

        sentiments_count = pd.Series(sentiments).value_counts()

        plt.figure(figsize=(8, 5))
        sns.barplot(x=sentiments_count.index, y=sentiments_count.values)
        plt.title('Sentiment Distribution')
        plt.savefig(os.path.join(self.metrics_folder, 'sentiment_distribution.png'))

        return sentiments_count, outcomes
