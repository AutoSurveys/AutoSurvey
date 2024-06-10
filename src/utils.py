import os
from typing import List
import threading
import tiktoken
from tqdm import trange
import time
import requests
import random
import json
from langchain.document_loaders import PyPDFLoader

class tokenCounter():

    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.model_price = {}
        
    def num_tokens_from_string(self, string:str) -> int:
        return len(self.encoding.encode(string))

    def num_tokens_from_list_string(self, list_of_string:List[str]) -> int:
        num = 0
        for s in list_of_string:
            num += len(self.encoding.encode(s))
        return num
    
    def compute_price(self, input_tokens, output_tokens, model):
        return (input_tokens/1000) * self.model_price[model][0] + (output_tokens/1000) * self.model_price[model][1]

    def text_truncation(self,text, max_len = 1000):
        encoded_id = self.encoding.encode(text, disallowed_special=())
        return self.encoding.decode(encoded_id[:min(max_len,len(encoded_id))])

def load_pdf(file, max_len = 1000):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    text = ''.join([p.page_content for p in pages])
    return encoding.decode(encoding.encode(text)[:max_len])