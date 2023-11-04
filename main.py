'''
chatbot service to extract information from text entered by user
'''
import json
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from src.chatbot_service import BotService
from src.validators import Request, Response

with open('configs.json', 'r') as f:
    CONFIG = json.load(f)

app = FastAPI()
service = BotService(CONFIG)

@app.post("/chat", response_model=Response)
def get_response(user_request: Request):
    '''
    endpoint for the main service
    '''
    json_request = jsonable_encoder(user_request)
    response = service.extract_information(json_request)
    return response
