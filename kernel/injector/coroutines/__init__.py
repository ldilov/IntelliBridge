import os
import importlib
from types import ModuleType
from typing import List


def import_coroutines(package_name: str) -> List[ModuleType]:
    package = importlib.import_module(package_name)
    package_dir = os.path.dirname(package.__file__)

    coroutines = []
    for file_name in os.listdir(package_dir):
        if not file_name.endswith(".py") or file_name == "__init__.py":
            continue

        module_name = file_name[:-3]  # Remove '.py' extension
        module = importlib.import_module(f"{package_name}.{module_name}")

        coroutines.append(module)

    return coroutines


coroutines = import_coroutines(__name__)
__all__ = ['coroutines']
