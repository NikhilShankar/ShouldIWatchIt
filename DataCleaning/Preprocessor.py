import os
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import emoji
from datetime import datetime

class CommentPreprocessor:
    def __init__(self, dataframe, save_folder):
        """
        Initialize the CommentPreprocessor class.

        :param dataframe: Input DataFrame containing the data to process
        :param save_folder: Folder where the cleaned DataFrame will be saved
        """
        self.dataframe = dataframe
        self.save_folder = save_folder
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def remove_stopwords(self, text):
        """Remove stopwords from the text."""
        return ' '.join([word for word in text.split() if word.lower() not in self.stopwords])

    def remove_emojis(self, text):
        """Remove emojis from the text."""
        return emoji.replace_emoji(text, "")

    def lemmatize_text(self, text):
        """Lemmatize words in the text."""
        return ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def preprocess_comments(self):
        """
        Preprocess the comments column by removing stopwords, removing emojis, and lemmatizing.
        """
        if 'Comments' in self.dataframe.columns:
            self.dataframe['Comments'] = self.dataframe['Comments'].astype(str)
            self.dataframe['Comments'] = self.dataframe['Comments'].apply(lambda x: x.lower())
            self.dataframe['Comments'] = self.dataframe['Comments'].apply(lambda x: re.sub(f"[{string.punctuation}]", "", x))
            self.dataframe['Comments'] = self.dataframe['Comments'].apply(self.remove_stopwords)
            self.dataframe['Comments'] = self.dataframe['Comments'].apply(self.remove_emojis)
            self.dataframe['Comments'] = self.dataframe['Comments'].apply(self.lemmatize_text)

    def calculate_score(self):
        """
        Calculate the score based on Sentiments and Imdb Rating.
        """
        if all(col in self.dataframe.columns for col in ['Sentiments', 'Imdb Rating', 'Positive', 'Negative', 'Neutral']):
            def compute_score(row):
                if row['Sentiments'] == 'Positive':
                    return row['Imdb Rating'] * 10 * row['Positive']
                elif row['Sentiments'] == 'Negative':
                    return -10 * row['Imdb Rating'] * row['Negative']
                elif row['Sentiments'] == 'Neutral':
                    return row['Imdb Rating'] * row['Neutral']
                return 0

            self.dataframe['Score'] = self.dataframe.apply(compute_score, axis=1)

    def save_cleaned_dataframe(self):
        """
        Save the cleaned DataFrame to the specified folder.
        """
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        output_path = os.path.join(self.save_folder, f'cleaned_dataframe_{timestamp}.csv')
        self.dataframe.to_csv(output_path, index=False)

    def process(self):
        """
        Run the full preprocessing pipeline.
        """
        self.preprocess_comments()
        self.calculate_score()
        self.save_cleaned_dataframe()
        return self.dataframe

