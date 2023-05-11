import multiprocessing
from tqdm.rich import tqdm_rich
from huggingface_hub import snapshot_download

from kernel.logger.logger import logger
from src.hub.abstract_hub import AbstractHub


class HuggingFaceHub(AbstractHub):
    def __init__(self):
        self.n_threads = int(multiprocessing.cpu_count() * 5)

    def download(self, repo_or_name, dir_name):
        logger.info(f"Downloading model from HuggingFace Hub: {repo_or_name} -> {dir_name}")

        snapshot_download(
            local_dir=dir_name,
            repo_id=repo_or_name,
            local_dir_use_symlinks=False,
            repo_type="model",
            tqdm_class=tqdm_rich,
            max_workers=1,
        )

        logger.success(f"Successfully downloaded model from HuggingFace Hub")