'''
service to get llm response
'''
import os
import replicate
from rebuff import Rebuff

class Assistant:
    '''
	sends request to llm and detects prompt injection
	'''
    def __init__(self, config):
        os.environ['REPLICATE_API_TOKEN'] = config['replicate_key']
        self.system_prompt = """You are a persuasive conversationalist for gathering information from 'User'.
        You must actively engage 'User' and encourage 'User' to share their information willingly.
        A NEGATIVE message indicates that you must take a proactive approach, initiating interactions and adeptly transitioning into small talk or different topics.
        Once 'User' sends a POSITIVE message, you have to seamlessly return to encouraging 'User' to share their information willingly.
        You must not let 'User' know that you are gathering information.
        You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."""
        self.rb = Rebuff(
            api_token=config["rebuff_token"],
            api_url="https://www.rebuff.ai"
        )

    def chat(self, message_history):
        '''
        sends conversation to llm and gets its response
        Paramters:
            message_history (List[dict]): Each dictionary contains previous messages with these keys
                role (str): 'User' or 'Assistant'
                content (str): message from 'User' or 'Assistant'
        Returns:
            str: response of llm
        '''
        messages = self.system_prompt
        for ele in message_history:
            messages += f"{ele['role']}: {ele['content']}\n\n"
        output = replicate.run(
            'meta/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1',
            input={
                "prompt": f"{messages} Assistant: ",
                "temperature": 0.9,
                "max_length": 64,
                # "top_p": 0.95
            }
        )
        response = ""
        for item in output:
            response += item
        response = response.strip()
        return response

    def detect_injection(self, user_input):
        '''
        detects prompt injection
        '''
        result = self.rb.detect_injection(user_input)
        return result.injectionDetected
