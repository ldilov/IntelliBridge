import torch
import gc


class TorchGC(object):
    def __init__(self, args):
        self.args = args

    def clear_torch_cache(self):
        gc.collect()

        if not self.args.cpu:
            torch.cuda.empty_cache()