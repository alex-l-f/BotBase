import requests
import json
import copy
import os
import time
from typing import List, Union, Dict, Optional
from transformers.models.auto.tokenization_auto import AutoTokenizer
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

class Conversation:
    """
    Keeps track of the conversation history between user and assistant.
    """
    def __init__(self, system_prompt: Optional[str] = None):
        self.history = []
        if system_prompt is not None:
            self.history = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, message: str):
        #if the last message was from the user, we can merge the two messages
        if len(self.history) > 0 and self.history[-1]["role"] == "user":
            self.history[-1]["content"] += message
        else:
            self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        #if the last message was from the assistant, we can merge the two messages
        if len(self.history) > 0 and self.history[-1]["role"] == "assistant":
            self.history[-1]["content"] += message
        else:
            self.history.append({"role": "assistant", "content": message})

    def add_assistant_block(self, data: Dict):
        #make sure the role is assistant
        if "role" not in data or data["role"] != "assistant":
            raise ValueError("The role must be 'assistant'")
        self.history.append(data)


    def add_tool_message(self, tool_call_id: str, tool_name: str, tool_response: str | Dict):
        #if the tool response is a json object, we need to convert it to a string
        if isinstance(tool_response, dict):
            tool_response = json.dumps(tool_response)
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": tool_response,
        })

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def append_history(self, history: List[Dict[str, str]]):
        self.history.extend(history)
    
    def get_formatted_history(self) -> List[Dict[str, str]]:
        return self.history

    def __deepcopy__(self, memo):
        new_conversation = Conversation(self.history[0]["content"])
        new_conversation.history = copy.deepcopy(self.history, memo)
        return new_conversation

    def copy(self):
        return self.__deepcopy__({})

    def clear_history(self):
        self.history = [self.history[0]]

class OpenRouter_Interface:
    """
    Interface for communicating with OpenRouter's API.
    """

    MODEL_TOKENIZER_MAP = {
        "qwen/qwq-32b-preview": "Qwen/QwQ-32B-Preview",
        "qwen/qvq-72b-preview": "Qwen/QVQ-72B-Preview",
        "deepseek/deepseek-chat": "deepseek-ai/DeepSeek-V3",
        "qwen/qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B",
        "google/gemini-2.0-flash-thinking-exp:free": "google/flan-t5-base",
        "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/llama-3.1-405b-instruct": "meta-llama/Llama-3.1-405B-Instruct",
        "google/gemma-4-31b-it": "google/gemma-4-31b-it"
    }

    MODEL_TOKEN_LIMIT_MAP = {
        "qwen/qwq-32b-preview": 32768,
        "qwen/qvq-72b-preview": 128000,
        "deepseek/deepseek-chat": 64000,
        "google/gemini-2.0-flash-thinking-exp:free": 40000,
        "meta-llama/llama-3.3-70b-instruct": 131072,
        "meta-llama/llama-3.1-405b-instruct": 32000,
        "qwen/qwen3-235b-a22b": 40960,
        "google/gemma-4-31b-it": 256000
    }

    def __init__(self, system_prompt: str, model: str = "google/gemma-4-31b-it", url: str = "http://localhost:8000/v1", reasoning: bool = True):
        """
        Initialize the VLLM Interface VIA the OpenAI API.

        Args:
            system_prompt (str): The system prompt to initialize the conversation.
            model (str): The model to use for the API. Defaults to "google/gemma-4-31b-it".
            url (str): The base URL for the OpenRouter API. Defaults to "http://localhost:8000/v1".
            reasoning (bool): Whether to enable reasoning capabilities. Defaults to True.
        """

        self.system_prompt = system_prompt
        self.model = model
        self.url = url
        self.reasoning = reasoning

        # Load environment variables from .env file
        if not load_dotenv():
            raise RuntimeError("dotenv could not load .env file. Please ensure it exists and is accessible.")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in environment variables. Please set it in your .env file.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Load appropriate tokenizer based on model
        tokenizer_name = self.MODEL_TOKENIZER_MAP.get(model, "gpt2")  # Default to GPT2 tokenizer if model not found
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def truncate_conversation(self, conversation: Conversation, max_tokens: int, margin : float = 0.9) -> Conversation:
        """
        Truncate conversation to fit within max_tokens while preserving the system message.
        Returns a new truncated conversation.
        """
        conv = conversation.copy()
        history = conv.get_history()
        
        # Count system message tokens
        system_tokens = self.count_tokens(history[0]["content"])
        
        # Reserve tokens for system message and some padding for the response
        remaining_tokens = max_tokens - system_tokens
        remaining_tokens = int(remaining_tokens * margin)
        
        if remaining_tokens <= 0:
            # If system message is too long, keep only system message
            return Conversation(history[0]["content"])
        
        # Start with system message
        truncated_history = [history[0]]
        current_tokens = system_tokens
        
        # Add messages from the end (most recent first) until we hit token limit
        for msg in reversed(history[1:]):
            msg_tokens = self.count_tokens(str(msg))
            if current_tokens + msg_tokens <= remaining_tokens:
                truncated_history.insert(1, msg)  # Insert after system message
                current_tokens += msg_tokens
            else:
                break
                
        conv.history = truncated_history
        return conv
    
    def get_json_object(self, conversations: Union[Conversation, List[Conversation]], json_schema: BaseModel, max_tokens: int = 2048, stop_sequences = None) -> Dict:
        """
        Get JSON object from VLLM.

        Args:
            conversations (Union[Conversation, List[Conversation]]): The conversation or list of conversations to process
            json_schema (BaseModel): Pydantic Class for constrained generation
        Returns:
            Dict: The JSON schema from the API
        Raises:
            requests.RequestException: If there's an error in the network request
            json.JSONDecodeError: If there's an error decoding the JSON response
        """
        output = self.get_completion(
            conversations=conversations,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            json_schema=json_schema,
            reasoning=False
        )
        if isinstance(output, list):
            output = output[0]
        if "json_schema" not in output:
            raise ValueError("Response does not contain 'json_schema' field")
        return output["json_schema"]
    
    def get_text_completion(self, conversations: Union[Conversation, List[Conversation]], max_tokens: int = 2048, stop_sequences = None) -> str:
        """
        Get text completion from VLLM.

        Args:
            conversations (Union[Conversation, List[Conversation]]): The conversation or list of conversations to process
            max_tokens (int): Maximum number of tokens to generate. Defaults to 2048

        Returns:
            str: The result or list of results from the API

        Raises:
            requests.RequestException: If there's an error in the network request
            json.JSONDecodeError: If there's an error decoding the JSON response
        """
        output = self.get_completion(
            conversations=conversations,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )
        if isinstance(output, list):
            output = output[0]
        return output["message"]
    
    def get_tools_completion(self, conversations: Union[Conversation, List[Conversation]], tools: List[Dict], max_tokens: int = 2048, stop_sequences = None) -> Dict:
        """
        Get completion with tool calls from VLLM.

        Args:
            conversations (Union[Conversation, List[Conversation]]): The conversation or list of conversations to process
            tools (List[Dict]): List of tools to use in the completion
            max_tokens (int): Maximum number of tokens to generate. Defaults to 2048

        Returns:
            Dict: The result or list of results from the API

        Raises:
            requests.RequestException: If there's an error in the network request
            json.JSONDecodeError: If there's an error decoding the JSON response
        """
        output = self.get_completion(
            conversations=conversations,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice="auto"
        )
        if isinstance(output, list):
            output = output[0]
        return output["message"], output["tools"]


    def get_completion(self, conversations: Union[Conversation, List[Conversation]], max_tokens: int = 512, stop_sequences = None, json_schema: Optional[BaseModel] = None, tools: List[Dict] = None, tool_choice: str = "none", reasoning = True) -> Dict:
        """
        Get completion from VLLM.

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
        for conversation in conversations:
            #get the model token limit
            model_token_limit = self.MODEL_TOKEN_LIMIT_MAP.get(self.model, 4096)
            # Truncate conversation if needed
            #truncated_conv = self.truncate_conversation(conversation, model_token_limit)  # Most models have 4096 context window

            max_retries = 5
            retry_delay = 1  # Start with 1 second delay
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=conversation.get_history(),
                        temperature=0.6,
                        top_p=0.95,
                        max_tokens=max_tokens,
                        #response_format={
                        #    "type": "json_schema", 
                        #    "json_schema": {
                        #        "name": json_schema.__class__.__name__,
                        #        "schema": json_schema.model_json_schema()
                        #    }
                        #} if json_schema else None,
                        tools=tools if tools else None,
                        tool_choice=tool_choice if tools else "none",
                        stop=stop_sequences,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": reasoning},
                            "guided_json": json_schema.model_json_schema() if json_schema else None,
                        }
                    )

                    if not response.choices:
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1}: Response missing 'choices' field, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            raise ValueError("Response missing 'choices' field after all retries")
                        
                    response_data = response.choices[0].message

                    #check if stop reason is length
                    if response.choices[0].finish_reason == "length":
                        #reattempt the request
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1}: Response stopped early due to length, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                    
                    result = {"message": response_data.content}

                    if json_schema:
                        result_json = json.loads(response_data.content)
                        result_model = json_schema.model_validate(result_json)
                        result["json_schema"] = result_model

                    if tools is not None:
                        tool_list = []
                        for i in range(len(response_data.tool_calls)):
                            response_data.tool_calls[i] = dict(response_data.tool_calls[i])
                            response_data.tool_calls[i]['function'] = dict(response_data.tool_calls[i]['function'])
                            tool = response_data.tool_calls[i]
                            tool_list.append({
                                "id": tool['id'],
                                "name": tool['function']['name'],
                                "arguments": json.loads(tool['function']['arguments']),
                            })
                        result["tools"] = tool_list

                    
                    conversation.add_assistant_block(dict(response_data))
                    results.append(result)
                    break  # Success, exit retry loop
                    
                except requests.RequestException as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}: Network error occurred: {e}, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    print(f"Network error occurred after all retries: {e}")
                    raise
                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}: Error decoding JSON response: {e}, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    print(f"Error decoding JSON response after all retries: {e}")
                    raise
                except ValueError as e:
                    print(f"Error in response format: {e}")
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}: Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    raise
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}: Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    raise

        return results[0] if len(results) == 1 else results

    def create_conversation(self) -> Conversation:
        """
        Create a new Conversation object with the current system prompt.

        Returns:
            Conversation: A new Conversation object.
        """
        return Conversation(self.system_prompt)
