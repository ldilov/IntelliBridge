from utils.quantizer.engine.enums.model_type import ModelType


def get_model_type(name_or_type: str) -> ModelType:
    gpt_based = ["gpt", "gpt2", "gpt3", "gptj", "gpt4all", "gpt4chan", "gpt4", "opt", "pygmalion"]
    lama_based = ["llama", "alpaca", "koala", "vicuna"]

    if any(s in name_or_type.lower() for s in gpt_based):
        return ModelType.GPTJ
    elif any(s in name_or_type.lower() for s in lama_based):
        return ModelType.LLAMA
    else:
        raise NotImplementedError(f'Quantization for {name_or_type} is not implemented yet.')