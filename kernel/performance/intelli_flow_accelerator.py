import math
import re
from typing import Dict

import accelerate
import nvidia_smi
import psutil
from transformers import PreTrainedModel, LlamaPreTrainedModel, LongformerPreTrainedModel, OPTPreTrainedModel, \
    RobertaPreTrainedModel, T5PreTrainedModel, GPTJPreTrainedModel, GPTNeoXPreTrainedModel, GPTNeoPreTrainedModel

from kernel.logger.logger import logger


class IntelliFlowAccelerator(object):
    def __init__(self):
        from kernel.persistence.memory.global_registry import registry as memory
        from tempfile import gettempdir

        self.args = memory.get('args')
        self.tmp_dir = gettempdir()
        self.no_split_modules = []

        self.no_split_modules.extend(LlamaPreTrainedModel._no_split_modules)
        self.no_split_modules.extend(LongformerPreTrainedModel._no_split_modules)
        self.no_split_modules.extend(OPTPreTrainedModel._no_split_modules)
        self.no_split_modules.extend(RobertaPreTrainedModel._no_split_modules)
        self.no_split_modules.extend(T5PreTrainedModel._no_split_modules)
        self.no_split_modules.extend(GPTJPreTrainedModel._no_split_modules)
        self.no_split_modules.extend(GPTNeoXPreTrainedModel._no_split_modules)
        self.no_split_modules.extend(GPTNeoPreTrainedModel._no_split_modules)

        self._cache = {}

    def accelerate(self, model: PreTrainedModel, checkpoint: str) -> PreTrainedModel:
        max_memory = self.get_max_memory()
        model = accelerate.load_checkpoint_and_dispatch(
            model,
            checkpoint,
            max_memory=max_memory,
            device_map="auto",
            offload_buffers=True,
            offload_folder=self.tmp_dir,
            offload_state_dict=True,
            no_split_module_classes=self.no_split_modules,
        )

        return model

    def get_memory_map(self, model):
        max_memory_map = self.get_max_memory()
        device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory_map,
                                                      no_split_module_classes=self.no_split_modules)

        return device_map

    def get_max_memory(self) -> Dict[str, str]:
        if self._cache.get('max_memory', False):
            return self._cache['max_memory']

        gpu_memory = self._get_gpu_memory()
        max_cpu_memory = str(self._get_cpu_memory())

        memory_map = list(map(lambda x: x.strip(), gpu_memory))

        max_memory = {}
        for i in range(len(memory_map)):
            max_memory[i] = f'{memory_map[i]}GB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
        max_memory['cpu'] = max_cpu_memory

        self._cache['max_memory'] = max_memory

        return max_memory

    def _get_gpu_memory(self):
        nvidia_smi.nvmlInit()

        count = nvidia_smi.nvmlDeviceGetCount()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        gpu_mem = count * [int(info.total / 1024 ** 3)]
        if len(gpu_mem) < 2:
            gpu_mem = [str(math.floor(0.5 * gpu_mem[0]))]
        else:
            gpu_mem = ' '.join(math.floor(0.5 * gpu_mem[0]))

        return gpu_mem

    def _get_cpu_memory(self):
        max_physical_memory = int(psutil.virtual_memory().total / 1024 ** 3)

        return f'{max_physical_memory}GB'
