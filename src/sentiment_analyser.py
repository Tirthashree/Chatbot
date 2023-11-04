'''
service to get sentiment of text
'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class SentimentAnalyser:
    '''
    Get sentiment of text
    '''
    def __init__(self, config):
        if config["sentiment"]["first_run"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                SENTIMENT_MODEL,
                cache_dir=config["sentiment"]["save_models_path"]["tokenizer"]
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                SENTIMENT_MODEL,
                cache_dir=config["sentiment"]["save_models_path"]["model"]
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                SENTIMENT_MODEL,
                cache_dir=config["sentiment"]["save_models_path"]["tokenizer"]
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config["sentiment"]["save_models_path"]["model"]
            )
        self.labels = {
            0: "NEGATIVE",
            1: "POSITIVE",
            2: "POSITIVE"
        }

    def get_sentiment(self, text):
        '''
        get sentiment of text
        Parameters:
            text (str): text to get sentiment
        Returns:
            str: NEGATIVE, NEUTRAL or POSITIVE
        '''
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        idx = np.argmax(output[0][0].detach().numpy())
        return self.labels[idx]
