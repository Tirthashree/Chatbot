'''
ner service to extract entities from text
'''
import os
import json
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

NER_MODEL  = "dslim/bert-base-NER"

def is_valid(entity):
    '''
    checks if extracted entity is not None or empty
    '''
    if not entity:
        return False
    return True

class NER:
    '''
    extracts entities from text
    '''
    def __init__(self, config):
        self.config = config["ner"]
        if self.config["first_run"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                NER_MODEL,
                cache_dir=self.config["save_models_path"]["tokenizer"]
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                NER_MODEL,
                cache_dir=self.config["save_models_path"]["model"]
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                NER_MODEL,
                cache_dir=self.config["save_models_path"]["tokenizer"]
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config["save_models_path"]["model"]
            )
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        if os.path.exists(self.config["db_path"]):
            with open(self.config["db_path"], "r") as f:
                self.database = json.load(f)
        else:
            self.database = {}
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
        self.phone_pattern = r'[\+\d{1,3}-]*\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'


    def _extract_phone(self, text):
        '''
        extract phone number from text
        '''
        phone_numbers = re.findall(self.phone_pattern, text)
        return phone_numbers

    def _extract_email(self, text):
        '''
        extract email from text
        '''
        email_addresses = re.findall(self.email_pattern, text)
        return email_addresses

    def extract_entities(self, text):
        '''
        extract entities like name, location organization, phone and email from text
        '''
        ner_results = self.nlp(text)
        phone = self._extract_phone(text)
        email = self._extract_email(text)
        entities = {}
        for ele in ner_results:
            if is_valid(ele["word"]) and ele["entity"] in self.config:
                entities[self.config[ele["entity"]]] = ele["word"]
        if is_valid(phone):
            entities["phone_number"] = phone
        if is_valid(email):
            entities["email_id"] = email
        return entities

    def store_data(self, user_id, data):
        '''
        to update and save user information in database
        Parameters:
            user_id (str): id of user
            data (dict): updated data of user
        '''
        self.database.update({user_id: data})
        self.save_database()

    def save_database(self):
        '''
        saves information in a database
        '''
        with open(self.config["db_path"], "w") as f:
            json.dump(self.database, f)
