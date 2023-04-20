from utils.services.model_service import ModelService
from utils.services.text_generation_service import TextGenerationService
from utils.configurator import Configurator
from kernel.arguments_parser import ArgumentsParser
from kernel.persistence.memory.global_modules_register import register as modules


class ApiService(object):
    def __init__(self):
        self.args_parser = modules.get(ArgumentsParser)
        self.args, self.args_defaults = (self.args_parser.args, self.args_parser.default_args)

        configurator = Configurator(self.args.model_dir)
        configurator.load_server_config()
        self.state_params = configurator.server_config

        # Services
        self.model_service = ModelService(self.args)
        model, tokenizer = self.model_service.load_model()
        self.text_generation_service = TextGenerationService(self.args, model, tokenizer)

    def generate(self, input_text):
        return self.text_generation_service.generate_reply(input_text, self.state_params)
