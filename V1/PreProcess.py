import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class PreProcess:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_comment(self, comment):
        print(comment)
        tokens = word_tokenize(comment.lower())
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in self.stop_words]
        return ' '.join(tokens)

    def preprocess_dataframe(self, dataframe):
        dataframe["Comment"] = dataframe["Comment"].apply(self.preprocess_comment)
        return dataframe
