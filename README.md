This is a FastAPI service for a chatbot that engages users in a conversation and encourages them to share their information willingly.\
\
It constantly analyses the sentiment of message from user using a [sentiment analysis model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).\
If the sentiment is NEGATIVE, implying that the user is not willing to disclose any information, the chatbot engages him in small talk and tries to reask the user.\
If the sentiment is POSITIVE, [an NER model](https://huggingface.co/dslim/bert-base-NER) is used to extract information like name, location and organization from the text.
Additionally, regex is used to extract phone number and email id from text.\
The extracted information is currently stored in a JSON file.\
\
The chatbot is powered by Llama-2 chat model via [Replicate](https://replicate.com/).\
\
In addition to the above functions, prompt injections are also detected using [Rebuff AI](https://www.rebuff.ai/).\
\
The service was created in Python 3.9.13.
To run the service:
1. Create a [Replicate token](https://replicate.com/).
2. Create a [Rebuff token](https://www.rebuff.ai/).
3. Replace these tokens in the respective fields in [configurations](configs.json).
4. Install the requirements:\
```pip install -r requirements.txt```
5. Run the following command in terminal:\
```uvicorn main:app```
6. Hit the endpoint: http://127.0.0.1:8000/chat \
Sample payload:
    ```javascript
        {
            "user_id": "abc",
            "message": "sample message",
            "clear": true
        }
    ```
