import glob
import json
import os
import pathlib
from pathlib import Path
from typing import Tuple, Union


class FileManager:
    instance = None

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)

        return cls.instance

    def read_file(self, file_path: Union[str, Path]) -> str:
        path = self.base_dir / file_path
        with path.open('r', encoding="utf-8") as file:
            content = file.read()
        return content

    def write_file(self, file_path: Union[str, Path], content: str) -> None:
        path = self.base_dir / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding="utf-8") as file:
            file.write(content)

    def create_dir(self, dir_path: Union[str, Path]) -> None:
        path = Path(os.path.join(self.base_dir, dir_path))
        path.mkdir(parents=True, exist_ok=True)

    def list_dir(self, dir_path: Union[str, Path]) -> Tuple[list[Path], list[Path]]:
        path = Path(os.path.join(self.base_dir, dir_path))
        entries = path.iterdir()

        files = [entry for entry in entries if entry.is_file()]
        dirs = [entry for entry in entries if entry.is_dir()]

        return files, dirs

    def get_file_paths(self, pattern: str) -> list[str]:
        return [str(path) for path in self.base_dir.glob(pattern)]

    def get_file_paths_recursively(self, pattern: str) -> list[str]:
        return [str(path) for path in self.base_dir.glob(f"**/{pattern}")]

    def delete_file(self, file_path: Union[str, Path]) -> None:
        path = self.base_dir / file_path
        if path.exists():
            path.unlink()

    def merge_into_json_file(self, src_dict: dict, file_dest_path: Union[str, Path]) -> None:
        dest_dict = json.loads(self.read_file(file_dest_path))
        result = {**dest_dict, **src_dict}
        self.write_file(file_dest_path, json.dumps(result, encoding="utf-8"))

    def exists(self, file_path: Union[str, Path]) -> bool:
        path = self.base_dir / file_path
        return path.exists()

    def rename_file(self, old_file_path: Union[str, Path], new_file_path: Union[str, Path]) -> None:
        old_path = self.base_dir / old_file_path
        new_path = self.base_dir / new_file_path
        old_path.rename(new_path)

    def get_file_ext(self, file_path: Union[str, Path]) -> str:
        return pathlib.Path(file_path).suffix

    def update_file_ext(self, file_path: Union[str, Path], new_ext: Union[str, Path]) -> str:
        old_ext = str(pathlib.Path(file_path).suffix)
        return file_path.replace(old_ext, new_ext)
