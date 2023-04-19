from abc import ABC

from transformers import AutoModelForCausalLM, AutoModel


class AbstractLoader(ABC):
    def __init__(self, name):
        if 'chatglm' in name.lower():
            self.loader_class = AutoModel
            self.trust_remote_code = True
        else:
            self.loader_class = AutoModelForCausalLM
            self.trust_remote_code = False

    def load(self):
        raise NotImplementedError("load() not implemented")
