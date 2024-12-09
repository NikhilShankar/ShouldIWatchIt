import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Training:
    def __init__(self, model, metrics_folder):
        self.model = model
        self.metrics_folder = metrics_folder

    def train(self, dataframe, preprocess, epochs=10, batch_size=32):
        dataframe = preprocess.preprocess_dataframe(dataframe)

        X_comments = dataframe["Comment"]
        X_score = dataframe["Score"]
        y_sentiments = pd.get_dummies(dataframe["Sentiment"]).values
        y_outcomes = pd.get_dummies(dataframe["Verdict"]).values

        X_comments_train, X_comments_val, X_score_train, X_score_val, y_sentiments_train, y_sentiments_val, y_outcomes_train, y_outcomes_val = train_test_split(
            X_comments, X_score, y_sentiments, y_outcomes, test_size=0.2, random_state=42
        )

        print(f"Shape TRAIN XCOMMENTS: {X_comments_train.shape}")
        print(f"Shape TRAIN XSCORE: {X_score_train.shape}")
        print(f"Shape TRAIN YSENTIMENTS: {y_sentiments_train.shape}")
        print(f"Shape TRAIN Y_OUTCOMES: {y_outcomes_train.shape}")

        checkpoint = ModelCheckpoint(
            os.path.join(self.metrics_folder, 'best_model.h5'), save_best_only=True, monitor='val_loss', mode='min', verbose=1
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

        #Tokenizing for training purpose
        tokenizer = Tokenizer(num_words=10000)  # Set a max number of words
        tokenizer.fit_on_texts(X_comments_train)

        X_comments_train_seq = tokenizer.texts_to_sequences(X_comments_train)
        X_comments_train_padded = pad_sequences(X_comments_train_seq, padding='post', maxlen=100)  # Adjust maxlen as needed

        X_comments_val_seq = tokenizer.texts_to_sequences(X_comments_val)
        X_comments_val_padded = pad_sequences(X_comments_val_seq, padding='post', maxlen=100)

        self.model.history = self.model.model.fit(
            X_comments_train_padded,
            [y_sentiments_train, y_outcomes_train],
            validation_data=(X_comments_val_padded, [y_sentiments_val, y_outcomes_val]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping]
        )
