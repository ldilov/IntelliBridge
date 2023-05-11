import importlib.util
import sys
from typing import Any
from kernel.logger.logger import logger


class Plugins:
    def __init__(self):
        self.plugins = {}
        self.repo = {
            "AutoGPTQ": {
                "path": "external.plugins.AutoGPTQ.auto_gptq.modeling",
                "author": "PanQiWei"
            }
        }

    def add_plugin(self, plugin_name):
        path = self.repo[plugin_name]["path"]
        author = self.repo[plugin_name]["author"]
        module: Any = importlib.import_module(path)
        sys.modules[plugin_name] = module
        self.plugins[plugin_name] = module

        if plugin_name in self.plugins:
            logger.success(f"Plugin '{author}/{plugin_name}' loaded successfully")

    def get_plugin(self, plugin_name):
        return self.plugins[plugin_name]


plugins = Plugins()
plugins.add_plugin("AutoGPTQ")

__all__ = ['plugins']
