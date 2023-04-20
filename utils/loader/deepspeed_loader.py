import os
import time
from pathlib import Path

import deepspeed
import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers.deepspeed import is_deepspeed_zero3_enabled, HfDeepSpeedConfig

from kernel.logger.logger import logger
from utils.attention.attention_modifier import AttentionModifier
from utils.loader.abstract_loader import AbstractLoader


class DeepSpeedLoader(AbstractLoader):
    def __init__(self, name, dir, nvme_offload_dir, bf16=False, xformers=False, sdp=False):
        super().__init__(name)

        self.name = name
        self.dir = dir
        self.isBf16 = bf16,
        self.nvme_offload_dir = nvme_offload_dir
        self.xformers = xformers
        self.sdp_attention = sdp

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        local_rank = 0
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed()
        self.ds_config = self._generate_ds_config(bf16, 1 * world_size, nvme_offload_dir)
        dschf = HfDeepSpeedConfig(self.ds_config)

    def load(self):
        t0 = time.time()

        model = self.loader_class.from_pretrained(Path(f"{self.dir}/{self.name}"), torch_dtype=torch.bfloat16 if self.isBf16 else torch.float16)
        model = deepspeed.initialize(model=model, config_params=self.ds_config, model_parameters=None, optimizer=None, lr_scheduler=None)[0]
        model.module.eval()  # Inference
        logger.warning(f"DeepSpeed ZeRO-3 is enabled: {is_deepspeed_zero3_enabled()}")

        if any((self.xformers, self.sdp_attention)):
            attn_modifier = AttentionModifier(self.xformers, self.sdp_attention)
            attn_modifier.apply_auto(model)

        # Loading the tokenizer
        if any((k in self.name.lower() for k in ['gpt4chan', 'gpt-4chan'])) and Path(f"{self.dir}/gpt-j-6B/").exists():
            tokenizer = AutoTokenizer.from_pretrained(Path(f"{self.dir}/gpt-j-6B/"))
        elif type(model) is transformers.LlamaForCausalLM:
            tokenizer = LlamaTokenizer.from_pretrained(Path(f"{self.dir}/{self.name}/"), clean_up_tokenization_spaces=True)
            # Leaving this here until the LLaMA tokenizer gets figured out.
            # For some people this fixes things, for others it causes an error.
            try:
                tokenizer.eos_token_id = 2
                tokenizer.bos_token_id = 1
                tokenizer.pad_token_id = 0
            except:
                pass
        else:
            tokenizer = AutoTokenizer.from_pretrained(Path(f"{self.dir}/{self.name}/"), trust_remote_code=self.trust_remote_code)

        logger.success(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
        return model, tokenizer

    def _generate_ds_config(self, ds_bf16, train_batch_size, nvme_offload_dir):
        if nvme_offload_dir:
            ds_config = {
                "fp16": {
                    "enabled": not ds_bf16,
                },
                "bf16": {
                    "enabled": ds_bf16,
                },
                "zero_optimization": {
                    "stage": 3,
                    "offload_param": {
                        "device": "nvme",
                        "nvme_path": nvme_offload_dir,
                        "pin_memory": True,
                        "buffer_count": 5,
                        "buffer_size": 1e9,
                        "max_in_cpu": 1e9
                    },
                    "overlap_comm": True,
                    "reduce_bucket_size": "auto",
                    "contiguous_gradients": True,
                    "sub_group_size": 1e8,
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "stage3_max_live_parameters": "auto",
                    "stage3_max_reuse_distance": "auto",
                },
                "aio": {
                    "block_size": 262144,
                    "queue_depth": 32,
                    "thread_count": 1,
                    "single_submit": False,
                    "overlap_events": True
                },
                "steps_per_print": 2000,
                "train_batch_size": train_batch_size,
                "train_micro_batch_size_per_gpu": 1,
                "wall_clock_breakdown": False
            }
        else:
            ds_config = {
                "fp16": {
                    "enabled": not ds_bf16,
                },
                "bf16": {
                    "enabled": ds_bf16,
                },
                "zero_optimization": {
                    "stage": 3,
                    "offload_param": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "stage3_max_live_parameters": "auto",
                    "stage3_max_reuse_distance": "auto",
                },
                "steps_per_print": 2000,
                "train_batch_size": train_batch_size,
                "train_micro_batch_size_per_gpu": 1,
                "wall_clock_breakdown": False
            }

        return ds_config
