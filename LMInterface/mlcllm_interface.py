import requests
import json
import copy
import os
from typing import List, Union, Dict, Optional
from pydantic import BaseModel
from threading import Thread
from queue import Queue

class Conversation:
    """
    Keeps track of the conversation history between user and assistant.
    """
    def __init__(self, system_prompt: str):
        self.history = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, message: str):
        #if the last message was from the user, we can merge the two messages
        if self.history[-1]["role"] == "user":
            self.history[-1]["content"] += message
        else:
            self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        #if the last message was from the assistant, we can merge the two messages
        if self.history[-1]["role"] == "assistant":
            self.history[-1]["content"] += message
        else:
            self.history.append({"role": "assistant", "content": message})

    def get_history(self) -> List[Dict[str, str]]:
        return self.history
    
    def get_formatted_history(self) -> str:
        formatted = ""
        for message in self.history:
            role = message["role"]
            content = message["content"]
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        return formatted

    def __deepcopy__(self, memo):
        new_conversation = Conversation(self.history[0]["content"])
        new_conversation.history = copy.deepcopy(self.history, memo)
        return new_conversation

    def copy(self):
        return self.__deepcopy__({})

    def clear_history(self):
        self.history = [self.history[0]]


def parallel_requests(urls: List[str], payloads: List[dict]) -> List[dict]:
    """
    Dispatches multiple requests in parallel and collects their results.
    Args:
        urls: List of URLs to send requests to
        payloads: List of payloads corresponding to each URL
    Returns:
        List of response dictionaries
    """
    threads = []
    result_queue = Queue()
    
    # Create and start threads
    for url, payload in zip(urls, payloads):
        thread = Thread(target=make_request, args=(url, payload, result_queue))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    return results

def make_request(url: str, payload: dict, result_queue: Queue):
    """
    Makes a POST request to the specified URL with the provided payload.
    Args:
        url: URL to send the request to
        payload: Payload to send with the request
        result_queue: Queue to store the response
    """
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result_queue.put(response.json())
    except requests.RequestException as e:
        print(f"Network error occurred: {e}")
        result_queue.put(None)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        result_queue.put(None)


class MLCLLM_Interface:
    """
    Interface for communicating with hosted 
    """

    def __init__(self, system_prompt: str):
        """
        Initialize the MLCLLM_Interface.

        Args:
            system_prompt (str): The system prompt to be used in all requests
        """
        self.system_prompt = system_prompt
        self.url = "http://hidamari.lan:8000/v1/chat/completions"

    def get_completion(self, conversations: Union[Conversation, List[Conversation]], max_tokens: int = 1024, json_schema: BaseModel = None) -> Union[str, List[str]]:
        """
        Get completion from MLC-LLM.

        Args:
            conversations (Union[Conversation, List[Conversation]]): The conversation or list of conversations to process
            max_tokens (int): Maximum number of tokens to generate. Defaults to 2048
            json_schema (BaseModel): Pydantic Class for constrained generation (Note: may not be supported by all models)

        Returns:
            Union[str, List[str]]: The result or list of results from the API

        Raises:
            requests.RequestException: If there's an error in the network request
            json.JSONDecodeError: If there's an error decoding the JSON response
        """
        if isinstance(conversations, Conversation):
            conversations = [conversations]

        results = []
        payloads = []
        for conversation in conversations:
            data = {
                "model": "./l8b_q4f16_mlc/",
                "messages": conversation.get_history(),
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object", "schema": json.dumps(json_schema.model_json_schema())} if json_schema else {"type": "text"}
            }


            payloads.append(data)

            # try:
            #     response = requests.post(
            #         url=self.url,
            #         json=data
            #     )
            #     response.raise_for_status()


        for response in parallel_requests([self.url] * len(conversations), payloads):
            raw_result = response["choices"][0]["message"]["content"]
            
            if json_schema:
                # Verify JSON response matches schema
                result_json = json.loads(raw_result)
                result = json_schema.model_validate(result_json)
            else:
                result = raw_result
            
            conversation.add_assistant_message(raw_result)
            results.append(result)

        return results[0] if len(results) == 1 else results

    def create_conversation(self) -> Conversation:
        """
        Create a new Conversation object with the current system prompt.

        Returns:
            Conversation: A new Conversation object.
        """
        return Conversation(self.system_prompt)

# Example usage:
# api_key = os.getenv("OPENROUTER_API_KEY")  # Get API key from environment variable
# system_prompt = "You are a helpful conversational assistant. Help the user with whatever they need."
# router = OpenRouterInterface(api_key, system_prompt)
# 
# # Single conversation example
# conv1 = router.create_conversation()
# conv1.add_user_message("Hello! Can you tell me about the weather today?")
# result1 = router.get_completion(conv1)
# print(result1)
# 
# # Multiple conversations example
# conv2 = router.create_conversation()
# conv2.add_user_message("What about tomorrow's forecast?")
# conv3 = router.create_conversation()
# conv3.add_user_message("And the day after?")
# results = router.get_completion([conv2, conv3])
# print(results)
