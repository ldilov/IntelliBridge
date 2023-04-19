import argparse
import re
import tempfile

import psutil


class ArgumentsParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54)
        )

        self.args = None
        self.default_args = None

    def parse(self):
        self.parser.add_argument('--chat', action='store_true', help='Launch the web UI in chat mode with a style similar to the Character.AI website.')
        self.parser.add_argument('--notebook', action='store_true', help='Launch the web UI in notebook mode, where the output is written to the same text box as the input.')
        self.parser.add_argument('--model', type=str, help='Name of the model to load by default.')
        self.parser.add_argument('--lora', type=str, help='Name of the LoRA to apply to the model by default.')
        self.parser.add_argument("--model-dir", type=str, default='models/', help="Path to directory with all the models")
        self.parser.add_argument("--lora-dir", type=str, default='loras/', help="Path to directory with all the loras")
        self.parser.add_argument('--model-menu', action='store_true', help='Show a model menu in the terminal when the web UI is first launched.')
        self.parser.add_argument('--no-stream', action='store_true', help='Don\'t stream the text output in real time.')
        self.parser.add_argument('--settings', type=str, help='Load the default interface settings from this json file. See settings-template.json for an example. If you create a file called settings.json, this file will be loaded by default without the need to use the --settings flag.')
        self.parser.add_argument('--extensions', type=str, nargs="+", help='The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.')
        self.parser.add_argument('--verbose', action='store_true', help='Print the prompts to the terminal.')

        # Accelerate/transformers
        self.parser.add_argument('--cpu', action='store_true', help='Use the CPU to generate text. Warning: Training on CPU is extremely slow.')
        self.parser.add_argument('--auto-devices', action='store_true', help='Automatically split the model across the available GPU(s) and CPU.')
        self.parser.add_argument('--gpu-memory', type=str, nargs="+", help='Maxmimum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values in MiB like --gpu-memory 3500MiB.')
        self.parser.add_argument('--cpu-memory', type=str, help='Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.')
        self.parser.add_argument('--disk-cache-dir', type=str, default="cache", help='Directory to save the disk cache to. Defaults to "cache".')
        self.parser.add_argument('--bf16', action='store_true', help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')
        self.parser.add_argument('--no-cache', action='store_true', help='Set use_cache to False while generating text. This reduces the VRAM usage a bit at a performance cost.')
        self.parser.add_argument('--xformers', action='store_true', help="Use xformer's memory efficient attention. This should increase your tokens/s.")
        self.parser.add_argument('--sdp-attention', action='store_true', help="Use torch 2.0's sdp attention.")

        # GPTQ
        self.parser.add_argument('--wbits', type=int, default=0, help='Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported.')
        self.parser.add_argument('--model_type', type=str, help='Model type of pre-quantized model. Currently LLaMA, OPT, and GPT-J are supported.')
        self.parser.add_argument('--groupsize', type=int, default=-1, help='Group size.')
        self.parser.add_argument('--pre_layer', type=int, default=0, help='The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models.')
        self.parser.add_argument('--monkey-patch', action='store_true', help='Apply the monkey patch for using LoRAs with quantized models.')
        self.parser.add_argument('--no-quant_attn', action='store_true', help='(triton) Disable quant attention. If you encounter incoherent results try disabling this.')
        self.parser.add_argument('--no-warmup_autotune', action='store_true', help='(triton) Disable warmup autotune.')
        self.parser.add_argument('--no-fused_mlp', action='store_true', help='(triton) Disable fused mlp. If you encounter "Unexpected mma -> mma layout conversion" try disabling this.')


# llama.cpp
        self.parser.add_argument('--threads', type=int, default=0, help='Number of threads to use in llama.cpp.')

        # DeepSpeed
        self.parser.add_argument('--deepspeed', action='store_true', help='Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration.')
        self.parser.add_argument('--nvme-offload-dir', type=str, help='DeepSpeed: Directory to use for ZeRO-3 NVME offloading.')
        self.parser.add_argument('--local_rank', type=int, default=0, help='DeepSpeed: Optional argument for distributed setups.')

        args = self.parser.parse_args()
        args_defaults = self.parser.parse_args([])

        self._normalize_args(args)

        self.args = args
        self.default_args = args_defaults

        print(f"Arguments: {args}")

        return args, args_defaults

    def is_chat(self):
        return self.args.chat

    def _normalize_args(self, args):
        args.wbits = self._get_wbits(args)
        args.groupsize = self._get_group_size(args)
        args.gpu_memory = self._get_gpu_memory(args)
        args.cpu_memory = self._get_cpu_memory(args)
        # args.monkey_patch = self._is_using_lora(args)

        args.no_quant_attn = True
        args.no_warmup_autotune = True
        args.no_fused_mlp = True

        if not self._is_quantized(args):
            args.deepspeed = True
            args.nvme_offload_dir = tempfile.gettempdir()

    def _is_quantized(self, args):
        return "gptq" in args.model.lower()   or\
                            args.wbits != 0   or \
                            args.monkey_patch or \
                            args.groupsize != -1

    def _get_wbits(self, args):
        found_wbits = re.search(r'(int(?P<wbits1>[48])|(?P<wbits2>[48])-?bit)', args.model, re.I)

        if found_wbits and args.wbits == 0:
            return int(found_wbits.group('wbits1') or found_wbits.group('wbits2'))

        return args.wbits

    def _is_using_lora(self, args):
        found_lora = re.search(r'lora', args.model, re.I)

        if found_lora and not args.monkey_patch:
            return True

        return args.monkey_patch

    def _get_group_size(self, args):
        match = re.search(r'((?P<group_size>\d{1,3})g|groupsize(?P<group_size2>\d{1,3}))', args.model, re.I)
        group_size = args.groupsize

        if match is not None and group_size == -1:
            group_size = int(match.group('group_size')) if match.groupdict()['group_size'] else int(match.group('group_size2'))

        return group_size

    def _get_gpu_memory(self, args):
        import nvidia_smi
        nvidia_smi.nvmlInit()

        count = nvidia_smi.nvmlDeviceGetCount()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        gpu_mem = count * [int(info.total / 1024**3)]
        if len(gpu_mem) < 2:
            gpu_mem = gpu_mem[0]
        else:
            gpu_mem = ' '.join(gpu_mem)

        return args.gpu_memory or gpu_mem

    def _get_cpu_memory(self, args):
        max_physical_memory = int(psutil.virtual_memory().total / 1024**3)

        return args.cpu_memory or max_physical_memory

    def _get_model_type(self, args):
        if "gptj" in args.model.lower():
            return "gptj"
        elif "opt" in args.model.lower():
            return "opt"
        elif "llama" in args.model.lower() or \
             "alpaca" in args.model.lower() or \
             "vicuna" in args.model.lower():
            return "llama"


        return args.cpu_memory or max_physical_memory
