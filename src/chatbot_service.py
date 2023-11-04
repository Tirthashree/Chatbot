'''
chatbot service to extract information from text entered by user
'''
import os
import json
import logging
from src.sentiment_analyser import SentimentAnalyser
from src.ner import NER
from src.assistant import Assistant

logger = logging.getLogger(__name__)
f_handler = logging.FileHandler('service.log')
logger.addHandler(f_handler)

class BotService:
    '''
    extracts information from text entered by user
    '''
    def __init__(self, config):
        self.analyser = SentimentAnalyser(config)
        self.ner_service = NER(config)
        self.assistant_service = Assistant(config)
        if os.path.exists(config["conversation_history_path"]):
            with open(config["conversation_history_path"], "r") as f:
                self.conversation_history = json.load(f)
        else:
            self.conversation_history = {}
        self.config = config

    def extract_information(self, request):
        '''
        identifies sentiment of text entered by user, extracts information from it \
            and responds back to user
        Parameters:
            request (dict):
                user_id (str)
                message (str) - message entered by user
                clear (bool) - a flag to clear user chat history
        Returns:
            dict:
                status (int)
                response (str) - response given by llm
        '''
        user_id = request["user_id"]
        user_message = request["message"]
        clear = request.get("clear")
        final_response = {
            "status": 200,
            "response": ""
        }
        try:
            flag = self.assistant_service.detect_injection(user_message)
            if flag:
                final_response["response"] = "Cannot generate a response for your query. Please try again."
                return final_response
            if clear:
                if user_id in self.conversation_history:
                    del self.conversation_history[user_id]
                user_history = []
            else:
                user_history = self.conversation_history.get(user_id, [])
            sentiment = self.analyser.get_sentiment(user_message)
            if sentiment:
                entities = self.ner_service.extract_entities(user_message)
                if entities:
                    self.ner_service.store_data(user_id, entities)
            msg = f"This is a {sentiment} message.\n\n{user_message}"
            user_history.append({"role": "User", "content": msg})
            llm_response = self.assistant_service.chat(user_history)
            user_history.append({"role": "Assistant", "content": llm_response})

            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = user_history
            final_response["response"] = llm_response
            self.save_history()
        except Exception as err_msg:
            logger.error(f"Encountered error: {err_msg}")
            final_response["status"] = 500
            final_response["response"] = "Internal Server Error"
        return final_response

    def save_history(self):
        '''
        helper function to save conversation history of users
        '''
        with open(self.config["conversation_history_path"], "w") as f:
            json.dump(self.conversation_history, f)
