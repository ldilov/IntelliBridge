from src.data_processing.loader.gptq_loader import GptQLoader


def load_model_llama_8bit(gptq_loader: GptQLoader):
    return gptq_loader.load()
