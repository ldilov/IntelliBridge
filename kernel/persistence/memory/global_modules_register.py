import importlib
import os

from kernel.arguments_parser import ArgumentsParser
from kernel.logger.logger import logger
from kernel.persistence.memory.global_register import GlobalRegister
from kernel.persistence.storage.file_manager import FileManager


class GlobalModulesRegister(GlobalRegister):
    def __init__(self):
        super().__init__()

    def register(self, module: type, value):
        super().register(module.__class__.__name__, value)

    def get(self, module: type):
        return super().get(module.__class__.__name__)

    def deregister(self, module: type):
        return super().deregister(module.__class__.__name__)

    def import_and_register_module(self, module_name, object_name=None):
        try:
            module_instance = importlib.import_module(module_name)
            if object_name:
                object_instance = getattr(module_instance, object_name)
                key = f"{module_name}.{object_instance}"
                value = object_instance
            else:
                key = module_name
                value = module_instance

            self.register(key, value)
            return value
        except ImportError:
            logger.error(f"Could not import module '{module_name}'")
        except AttributeError:
            logger.error(f"Error: Could not find object '{object_name}' in module '{module_name}'")


# Initialize global modules
filemanager_instance = FileManager(os.getcwd())
argumentsparser_instance = ArgumentsParser()
argumentsparser_instance.parse()

register = GlobalModulesRegister()
register.register(FileManager, filemanager_instance)
register.register(ArgumentsParser, argumentsparser_instance)

__all__ = ['register']
