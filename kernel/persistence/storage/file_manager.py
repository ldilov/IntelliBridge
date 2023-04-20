import glob
import pathlib
from pathlib import Path


class FileManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def read_file(self, file_path: str) -> str:
        path = self.base_dir / file_path
        with path.open('r') as file:
            content = file.read()
        return content

    def write_file(self, file_path: str, content: str) -> None:
        path = self.base_dir / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as file:
            file.write(content)

    def get_file_paths(self, pattern: str) -> list[str]:
        return [str(path) for path in self.base_dir.glob(pattern)]

    def get_file_paths_recursively(self, pattern: str) -> list[str]:
        return [str(path) for path in self.base_dir.glob(f"**/{pattern}")]

    def delete_file(self, file_path: str) -> None:
        path = self.base_dir / file_path
        if path.exists():
            path.unlink()

    def rename_file(self, old_file_path: str, new_file_path: str) -> None:
        old_path = self.base_dir / old_file_path
        new_path = self.base_dir / new_file_path
        old_path.rename(new_path)

    def get_file_ext(self, file_path: str) -> str:
        return pathlib.Path(file_path).suffix

    def update_file_ext(self, file_path: str, new_ext: str) -> str:
        old_ext = str(pathlib.Path(file_path).suffix)
        return file_path.replace(old_ext, new_ext)
