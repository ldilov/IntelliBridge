import re
from pathlib import Path

import accelerate
import psutil
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from utils.loader.abstract_loader import AbstractLoader
from utils.third_party.gptq.llama.llama_inference_offload import make_quant as gptq_make_quant, load_quant as gptq_load_quant
from utils.third_party.gptq.llama.modelutils import find_layers as gptq_find_layers

from utils.third_party.gptq.gptj.gptj_inference import load_quant as gptj_load_quant
from utils.third_party.gptq.gptj.modelutils import find_layers as gptj_find_layers


class GptQLoader(AbstractLoader):
    def __init__(self, args):
        super().__init__(args.model)
        self.args = args

    def load(self):
        if not self.args.model_type:
            name = self.args.model.lower()
            if any((k in name for k in ['llama', 'alpaca', 'vicuna'])):
                model_type = 'llama'
            elif any((k in name for k in ['opt-', 'galactica'])):
                model_type = 'opt'
            elif any((k in name for k in ['gpt-j', 'pygmalion-6b', 'gpt-neo', 'gpt4'])):
                model_type = 'gptj'
            else:
                print("Can't determine model type from model name. Please specify it manually using --model_type "
                      "argument")
                exit()
        else:
            model_type = self.args.model_type.lower()

        # Select the appropriate load_quant function
        if model_type == 'llama':
            load_quant = gptq_load_quant
        elif model_type in ('opt', 'gptj'):
            load_quant = gptj_load_quant
        else:
            load_quant = self._load_quant

        # Find the quantized model weights file (.pt/.safetensors)
        path_to_model = Path(f'{self.args.model_dir}/{self.args.model}')
        pt_path = self._find_quantized_model_file(self.args.model)
        if not pt_path:
            print("Could not find the quantized model in .pt or .safetensors format, exiting...")
            exit()
        else:
            print(f"Found the following quantized model: {pt_path}")

        extension = Path(path_to_model).suffix
        path_to_model = str(path_to_model).replace(extension, ".config.json")
        if model_type == 'llama':
            model = load_quant(str(path_to_model), str(pt_path), self.args.wbits, self.args.groupsize, self.args.pre_layer)
        elif model_type in ('opt', 'gptj'):
            model = load_quant(str(path_to_model), str(pt_path), self.args.wbits)
        else:
            print("Not llama/gpt based model! Using default load_quant function...")

            threshold = False if model_type == 'gptj' else 128
            model = load_quant(str(path_to_model), str(pt_path), self.args.wbits, self.args.groupsize, kernel_switch_threshold=threshold)

            # accelerate offload (doesn't work properly)
            if self.args.gpu_memory or torch.cuda.device_count() > 1:
                from tempfile import gettempdir

                if self.args.gpu_memory:
                    memory_map = list(map(lambda x: x.strip(), self.args.gpu_memory))
                    max_physical_memory = str(int(psutil.virtual_memory().total / 1024 / 1024 / 1024))
                    max_cpu_memory = self.args.cpu_memory.strip() if self.args.cpu_memory is not None else max_physical_memory
                    max_memory = {}
                    for i in range(len(memory_map)):
                        max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
                    max_memory['cpu'] = max_cpu_memory
                else:
                    max_memory = accelerate.utils.get_balanced_memory(model)

                device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory,  no_split_module_classes=["LlamaDecoderLayer"])
                print("Using the following device map for the quantized model:", device_map)
                model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True, offload_dir=gettempdir())

            # No offload
            elif not self.args.cpu:
                model = model.to(torch.device('cuda:0'))

        return model

    def _load_quant(self, model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=['lm_head'], kernel_switch_threshold=128, eval=True):
        print("Loading quantized model with default strategy...")

        def noop(*args, **kwargs):
            pass

        config = AutoConfig.from_pretrained(model)
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = AutoModelForCausalLM.from_config(config)
        torch.set_default_dtype(torch.float)
        if eval:
            model = model.eval()
        layers = gptq_find_layers(model)
        for name in exclude_layers:
            if name in layers:
                del layers[name]

        gptq_make_quant(model, layers, wbits, groupsize)

        del layers

        print('Loading model ...')
        if checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file as safe_load
            model.load_state_dict(safe_load(checkpoint), strict=False)
        else:
            model.load_state_dict(torch.load(checkpoint), strict=False)

        model.seqlen = 2048
        print('Done.')

        return model

    def _find_quantized_model_file(self, model_name):
        path_to_model = Path(f'{self.args.model_dir}/{model_name}')
        pt_path = None
        priority_name_list = [
            Path(f'{self.args.model_dir}/{model_name}{hyphen}{self.args.wbits}bit{group}{ext}')
            for group in ([f'-{self.args.groupsize}g', ''] if self.args.groupsize > 0 else [''])
            for ext in ['.safetensors', '.pt']
            for hyphen in ['-', f'/{model_name}-', '/']
        ]

        priority_name_list.append(path_to_model)

        for path in priority_name_list:
            if path.exists():
                pt_path = path
                break

        if not pt_path:
            found_pts = list(path_to_model.glob("*.pt"))
            found_safetensors = list(path_to_model.glob("*.safetensors"))
            pt_path = None

            if len(found_pts) > 0:
                if len(found_pts) > 1:
                    print('Warning: more than one .pt model has been found. The last one will be selected. It could be wrong.')
                pt_path = found_pts[-1]
            elif len(found_safetensors) > 0:
                if len(found_pts) > 1:
                    print('Warning: more than one .safetensors model has been found. The last one will be selected. It could be wrong.')
                pt_path = found_safetensors[-1]

        return pt_path
