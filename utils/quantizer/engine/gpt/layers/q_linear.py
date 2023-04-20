import os
import math
import torch

from utils.quantizer.engine.shared.q_abstract_linear_layer import QAbstractLinearLayer


class QLinear(QAbstractLinearLayer):
    def __init__(self, bits, groupsize, infeatures, outfeatures):
        super().__init__()

        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else infeatures

        self._initialize_buffers_and_variables()

    def _validate_bits(self, bits):
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    def _initialize_buffers_and_variables(self):
        self.register_buffer('qzeros', torch.zeros((math.ceil(self.infeatures/self.groupsize), self.outfeatures // 256 * (self.bits * 8)), dtype=torch.int))
        self.register_buffer('scales', torch.zeros((math.ceil(self.infeatures/self.groupsize), self.outfeatures)))
        self.register_buffer('bias', torch.zeros(self.outfeatures))
        self.register_buffer('qweight', torch.zeros((self.infeatures // 256 * (self.bits * 8), self.outfeatures), dtype=torch.int))

    def pack(self, linear, scales, zeros):
        super().pack(linear, scales, zeros)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        global quant_cuda

        outshape = list(input.shape)
        input = input.reshape(-1, input.shape[-1])
        output = self.bias.clone().repeat(input.shape[0], 1)
        outshape[-1] = self.bias.numel()
        dtype = input.dtype
        input = input.float()

        if self.bits == 2:
            quant_cuda.vecquant2matmul(input, self.qweight, output, self.scales.float(), self.qzeros, self.groupsize)
        elif self.bits == 3:
            quant_cuda.vecquant3matmul(input, self.qweight, output, self.scales.float(), self.qzeros, self.groupsize)
        elif self.bits == 4:
            quant_cuda.vecquant4matmul(input, self.qweight, output, self.scales.float(), self.qzeros, self.groupsize)
        elif self.bits == 8:
            quant_cuda.vecquant8matmul(input, self.qweight, output, self.scales.float(), self.qzeros, self.groupsize)

        output = output.to(dtype)
        output = output.reshape(outshape)

        return output

    def _prepare_intweight(self, linear, scale_zeros):
        intweight = []

        for idx in range(self.infeatures):
            g_idx = idx // self.groupsize
            intweight.append(torch.round((linear.weight.data[:,idx] + scale_zeros[g_idx]) / self.scales[g_idx]).to(torch.int)[:,None])

        return torch.cat(intweight, dim=1)

    def _pack_qweight(self, intweight):
        return super().pack_qweight_blocks(intweight, block_size=256, alpha=8)

    def _pack_qzeros(self, zeros):
        return super().pack_qzeros_blocks(zeros, block_size=256, alpha=8)