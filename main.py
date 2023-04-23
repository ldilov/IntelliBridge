import os

import torch

from kernel.persistence.infra.models.gptq_model import GptqModel
from utils.hub.huggingface import HuggingFaceHub
from utils.services.api_service import ApiService

api_service = ApiService()
for chunk in api_service.generate("Give me information about Elon Musk."):
    print(chunk)


# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-base-alpha-7b")
# model = AutoModelForCausalLM.from_pretrained("StabilityAI/stablelm-base-alpha-7b")
# model.half().cuda()
