from kernel.logger.logger import logger
from kernel.persistence.infra.models.ggml_model import GgmlModel


class LlamaCppGenerator(object):
    @classmethod
    def generate(cls, model: GgmlModel, input: str, state_params):
        generate_params = {}

        for k in ['temperature', 'top_p', 'top_k', 'repetition_penalty']:
            generate_params[k] = state_params[k]
        generate_params['token_count'] = state_params['max_new_tokens']
        try:
            for reply in model.generate(input, **generate_params):
                yield reply
        except Exception as ex:
            logger.error("Error in LlamaCppGenerator.generate: %s")