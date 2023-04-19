import json
import os
from pathlib import Path

import yaml


class Configurator(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_config = None
        self.server_config = None

    def load_model_config(self, name=None):
        # Loading model-specific settings (default)
        with Path(f'{self.model_dir}/config.yaml') as p:
            if p.exists():
                self.model_config = yaml.safe_load(open(p, 'r').read())
            else:
                self.model_config = {}

        file_name = f'{self.model_dir}/config-user.yaml'

        if name is not None:
            file_name = f'{self.model_dir}/{name}.yaml'

        # Applying user-defined model settings
        with Path(file_name) as p:
            if p.exists():
                user_config = yaml.safe_load(open(p, 'r').read())
                for k in user_config:
                    if k in self.model_config:
                        self.model_config[k].update(user_config[k])
                    else:
                        self.model_config[k] = user_config[k]

    def load_server_config(self):
        server_json_path = os.path.join(os.getcwd(), "config", "server.json")
        with Path(server_json_path) as pth:
            if pth.exists():
                self.server_config = json.loads(open(pth, 'r', encoding="utf-8").read())
            else:
                print(f"Server config file not found: {server_json_path}")
                self.server_config = {}

