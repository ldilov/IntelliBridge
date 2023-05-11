import json
import os
import re
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from kernel.persistence.memory.global_modules_registry import registry
from kernel.persistence.storage.file_manager import FileManager
from kernel.logger.logger import logger
from src.hub.abstract_hub import AbstractHub


class ModelStore:
    def __init__(self):
        self.file_manager: FileManager = registry.get(FileManager)
        self.base_path: Path = Path(os.getcwd()) / 'resources' / 'models'

    def store(self, hub: AbstractHub, repo_or_name: str, model_cls):
        model_path: Path = self._init_directory(repo_or_name)
        self._download(hub, model_path, repo_or_name)
        self._write_metadata(model_path, model_cls)

        if getattr(model_cls, 'generation_config', None) is not None:
            self._write_generation_config(model_path, model_cls.generation_config)
        else:
            self._write_generation_config(model_path)

    def load(self, name):
        import importlib
        model_path = self.base_path / name

        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        metadata = self._load_metadata(model_path)

        module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', metadata['model_class']).lower()
        module = importlib.import_module(f"kernel.persistence.infra.models.{module_name}")
        model_class = getattr(module, metadata['model_class'])
        model_module = model_class(model, tokenizer, metadata)
        model_module.generation_config = self._load_generation_config(model_path)

        return model_module, model_path

    def load_with_metadata_only(self, name):
        import importlib

        model_path = self.base_path / name
        metadata = self._load_metadata(model_path)

        module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', metadata['model_class']).lower()
        module = importlib.import_module(f"kernel.persistence.infra.models.{module_name}")
        model_class = getattr(module, metadata['model_class'])
        model_module = model_class(None, None, metadata)

        return model_module, model_path

    def save_config(self, name, config_name: str, config: dict[str, str]):
        if not self.file_manager.exists(str(self.base_path / name / f"{config_name}.json")):
            self.file_manager.write_file(str(self.base_path / name / f"{config_name}.json"), json.dumps(config))

    def load_config(self, name, config_name: str) -> dict:
        config = json.loads(self.file_manager.read_file(str(self.base_path / name / f"{config_name}.json")))
        return config

    def _load_metadata(self, model_path: Path) -> dict:
        metadata = json.loads(self.file_manager.read_file(str(model_path / "index.json")))
        return metadata

    def load_generation_config(self, model_path: Path):
        return self.load_config(model_path.name, "generation_config")

    def _write_generation_config(self, model_path: Path, config_version: str = "default"):
        model_name = model_path.name
        config_src = Path(os.getcwd()) / "config" / "generation" / f"generation.{config_version}.json"
        config_src_dict = json.loads(self.file_manager.read_file(str(config_src)), encoding="utf-8")
        self.save_config(model_name, "generation_config", config_src_dict)

    def _write_metadata(self, model_path, model_cls):
        files, dirs = self.file_manager.list_dir(str(model_path))
        model_name = Path(model_path).name
        extension = Path(model_path).suffix

        config = None
        tokenizer = None
        for file in files:
            file_path = Path(file)

            if not config and file_path.name in ["config.json", f"{model_name}.json", "model.json"]:
                config = file_path
            elif not tokenizer and file_path.name in ["tokenizer_config.json"]:
                tokenizer = file_path

        model_type = self._get_model_type(model_name, config)

        metadata = {
            "name": model_name,
            "config": str(config),
            "tokenizer": str(tokenizer),
            "model_type": model_type,
            "extension": extension,
            "model_class": model_cls.__name__
        }

        self.file_manager.write_file(model_path / "index.json", json.dumps(metadata))

    def _get_model_type(self, model_name, config_path=None):
        if config_path:
            config = json.loads(self.file_manager.read_file(str(config_path)))
            if config.get("model_type", False):
                return config.get("model_type")

        logger.warning(f"Could not find model type for model '{model_name}'. Using generic type!")

        return model_name

    def _init_directory(self, repo_or_name) -> Path:
        model_name = repo_or_name
        if "/" in repo_or_name or "\\" in repo_or_name:
            model_name = Path(model_name).name

        model_path = self.base_path / model_name
        self.file_manager.create_dir(str(model_path))
        return model_path

    def _download(self, hub, dir_name, repo_or_name):
        hub.download(repo_or_name, dir_name)
