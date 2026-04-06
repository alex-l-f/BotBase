import os
import openai
from openai import OpenAI, NOT_GIVEN
from typing import List, Union, Dict, Optional
import json
import numpy as np
from tqdm import tqdm
import logging
from openai import AsyncOpenAI
import asyncio
from asyncio import Semaphore

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

    def get_formatted_history(self) -> List[Dict[str, str]]:
        return self.history
    
    def append_history(self, history: List[Dict[str, str]]):
        self.history.extend(history)

    def __deepcopy__(self, memo):
        new_conversation = Conversation(self.history[0]["content"])
        new_conversation.history = self.history.copy()
        return new_conversation

    def copy(self):
        return self.__deepcopy__({})

    def clear_history(self):
        self.history = [self.history[0]]

    def print_history(self):
        for message in self.history:
            print(f"{message['role'].capitalize()}: {message['content']}")

    #to string method
    def __str__(self):
        return "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in self.history])

class OpenAI_Interface:
    """
    Interface for communicating with the OpenAI API.
    """

    def __init__(self, api_key: str, system_prompt: str):
        """
        Initialize the OpenAI_Interface.

        Args:
            api_key (str): The OpenAI API key.
            system_prompt (str): The system prompt to be used in all requests.
        """

        self.client = OpenAI(api_key=api_key)
        self.asyncclient = AsyncOpenAI(api_key=api_key)
        self.system_prompt = system_prompt
        self.sem = Semaphore(10)
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)

    async def send_query(self, complation_args):
        async with self.sem:
            try:
                completion = await self.asyncclient.beta.chat.completions.parse(**complation_args)
                if completion.choices[0].message.parsed is None:
                    result = completion.choices[0].message.content
                else:
                    result = completion.choices[0].message.parsed
                return result
            except openai.APIError as e:
                #Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                raise
            except openai.APIConnectionError as e:
                #Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                raise
            except openai.RateLimitError as e:
                #Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                raise


    async def get_async_completion(self, conversations: Union[Conversation, List[Conversation]], n_predict: Optional[int] = None, json_schema = None, stop_sequences = None) -> Union[str, List[str]]:
        """
        Get completion from the OpenAI API.

        Args:
            conversations (Union[Conversation, List[Conversation]]): The conversation or list of conversations to process.
            n_predict (int): The maximum number of tokens to generate. Defaults to 2048.
            json_schema (Optional[Dict]): JSON schema for constrained generation. Defaults to None.

        Returns:
            Union[str, List[str]]: The result or list of results from the API.

        Raises:
            openai.error.OpenAIError: If there's an error in the API request.
        """
        if isinstance(conversations, Conversation):
            conversations = [conversations]

        results = []
        arg_list = []
        for conversation in conversations:
            messages = conversation.get_formatted_history()
            
            completions_args = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.0,}
            if n_predict is not None:
                completions_args["max_tokens"] = n_predict
            if json_schema is not None:
                completions_args["response_format"] = json_schema
            if stop_sequences is not None:
                completions_args["stop"] = stop_sequences
            arg_list.append(completions_args)

        tasks = [asyncio.create_task(self.send_query(arg)) for arg in arg_list]
        completions = await asyncio.gather(*tasks)

        for i, completion in enumerate(completions):
            results.append(completion)
            #if the result is a pydantic class, we can convert it to a json string
            if hasattr(completion, "model_dump"):
                #convert the pydantic class into a json string
                result_json = json.dumps(completion.model_dump())
                conversations[i].add_assistant_message(result_json)
            else:
                #add the result as a string
                conversations[i].add_assistant_message(completion)

        return results[0] if len(results) == 1 else results

    def get_completion(self, conversations: Union[Conversation, List[Conversation]], n_predict: Optional[int] = None, json_schema = None, stop_sequences = None) -> Union[str, List[str]]:
        """
        Get completion from the OpenAI API.

        Args:
            conversations (Union[Conversation, List[Conversation]]): The conversation or list of conversations to process.
            n_predict (int): The maximum number of tokens to generate. Defaults to 2048.
            json_schema (Optional[Dict]): JSON schema for constrained generation. Defaults to None.

        Returns:
            Union[str, List[str]]: The result or list of results from the API.

        Raises:
            openai.error.OpenAIError: If there's an error in the API request.
        """
        if isinstance(conversations, Conversation):
            conversations = [conversations]

        results = []
        for conversation in conversations:
            messages = conversation.get_formatted_history()
            
            try:
                completions_args = {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "temperature": 0.0,}
                if n_predict is not None:
                    completions_args["max_tokens"] = n_predict
                if json_schema is not None:
                    completions_args["response_format"] = json_schema
                if stop_sequences is not None:
                    completions_args["stop"] = stop_sequences
                completion = self.client.beta.chat.completions.parse(**completions_args)
                if completion.choices[0].message.parsed is None:
                    result = completion.choices[0].message.content
                else:
                    result = completion.choices[0].message.parsed
                results.append(result)
                #if the result is a pydantic class, we can convert it to a json string
                if hasattr(result, "model_dump"):
                    #convert the pydantic class into a json string
                    result_json = json.dumps(result.model_dump())
                    conversation.add_assistant_message(result_json)
                else:
                    #add the result as a string
                    conversation.add_assistant_message(result)
            
            except openai.APIError as e:
                #Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                raise
            except openai.APIConnectionError as e:
                #Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                raise
            except openai.RateLimitError as e:
                #Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                raise

        return results[0] if len(results) == 1 else results

    def create_conversation(self) -> Conversation:
        """
        Create a new Conversation object with the current system prompt.

        Returns:
            Conversation: A new Conversation object.
        """
        return Conversation(self.system_prompt)

    def get_embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """
        Get embeddings for a list of texts using OpenAI's embedding model.

        Args:
            texts (List[str]): List of texts to get embeddings for.
            model (str): The name of the embedding model to use. Defaults to "text-embedding-3-small".

        Returns:
            List[List[float]]: A list of embedding vectors.

        Raises:
            openai.error.OpenAIError: If there's an error in the API request.
        """
        try:
            #batching only supports 2048 entries at once, so do it in groups
            embeddings = []
            #if we have over 4096 entries, show progress bar
            option_tqdm = tqdm if len(texts) > 4096 else lambda x: x
            for i in option_tqdm(range(0, len(texts), 2048)):
                end_len = min(i+2048, len(texts))
                response = self.client.embeddings.create(input=texts[i:end_len], model=model)
                embeddings.extend([embedding.embedding for embedding in response.data])
            return embeddings
        except openai.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            raise
        except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            raise
        except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            raise

# Example usage:
# api_key = os.getenv("OPENAI_API_KEY")
# system_prompt = "You are a helpful conversational assistant. Help the user with whatever they need."
# openai_interface = OpenAI_Interface(api_key, system_prompt)
# conv1 = openai_interface.create_conversation()
# conv1.add_user_message("Hello! Can you tell me about the weather today?")
# result1 = openai_interface.get_completion(conv1)
# print(result1)
# conv2 = openai_interface.create_conversation()
# conv2.add_user_message("What about tomorrow's forecast?")
# conv3 = openai_interface.create_conversation()
# conv3.add_user_message("And the day after?")
# results = openai_interface.get_completion([conv2, conv3])
# print(results)
# 
# # Example of using embeddings
# texts = ["Hello, world!", "OpenAI is amazing"]
# embeddings = openai_interface.get_embeddings(texts)
# print(f"Embeddings shape: {np.array(embeddings).shape}")