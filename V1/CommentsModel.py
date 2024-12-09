from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from keras.optimizers import Adam
import numpy as np

#Created with help from this tutorial and chatgpt
#https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

class CommentsModel:
    def __init__(self, unique_name):
        self.unique_name = unique_name
        self.model = None

    def build_model(self, vocab_size, embedding_dim, lstm_units):
        comment_input = Input(shape=(None,), name="Comment_Input")
        #imdb_input = Input(shape=(1,), name="Feature_Input")

        x = Embedding(vocab_size, embedding_dim)(comment_input)
        x = LSTM(lstm_units, return_sequences=False)(x)
        x = Dropout(0.5)(x)

        sentiment_output = Dense(3, activation="softmax", name="Sentiment_Output")(x)

        #combined = Concatenate()([x, imdb_input])
        combined = Dense(64, activation="relu", name="Dense1")(x)#(combined)
        combined = Dropout(0.5)(combined)
        outcome_output = Dense(3, activation="softmax", name="Outcome_Output")(combined)

        self.model = Model(inputs=comment_input, outputs=[sentiment_output, outcome_output])
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={"Sentiment_Output": "categorical_crossentropy", "Outcome_Output": "categorical_crossentropy"},
            metrics={"Sentiment_Output": "accuracy", "Outcome_Output": "accuracy"}
        )
        print(self.model.summary())
