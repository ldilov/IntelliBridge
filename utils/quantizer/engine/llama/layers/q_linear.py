import math
import numpy as np
import torch

from utils.quantizer.engine.shared.q_abstract_linear_layer import QAbstractLinearLayer


class QLinear(QAbstractLinearLayer):
    def __init__(self, bits, groupsize, infeatures, outfeatures, bias, kernel_switch_threshold=128, is_cuda=False):
        super().__init__()

        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        self.is_cuda = is_cuda
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.maxq = 2 ** self.bits - 1
        self.kernel_switch_threshold = kernel_switch_threshold

        self._initialize_buffers_and_variables(bias)

    def _validate_bits(self, bits):
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    def _initialize_buffers_and_variables(self, bias):
        self.register_buffer('qweight',
                             torch.zeros((self.infeatures // 32 * self.bits, self.outfeatures), dtype=torch.int32))
        self.register_buffer('qzeros',
                             torch.zeros(
                                 (math.ceil(self.infeatures / self.groupsize), self.outfeatures // 32 * self.bits),
                                 dtype=torch.int32))
        self.register_buffer('scales',
                             torch.zeros((math.ceil(self.infeatures / self.groupsize), self.outfeatures),
                                         dtype=torch.float16))
        self.register_buffer('g_idx',
                             torch.tensor([i // self.groupsize for i in range(self.infeatures)], dtype=torch.int32))
        if bias:
            self.register_buffer('bias', torch.zeros((self.outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.register_buffer('wf', torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0),
                                 persistent=False)
        elif self.bits == 3:
            self.register_buffer('wf', torch.tensor([[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                                                     [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                                                     [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0], ],
                                                    dtype=torch.int32).reshape(1, 3, 12), persistent=False)

    def pack(self, linear, scales, zeros, g_idx=None):
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        super().pack(linear, scales, zeros)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        global quant_cuda

        out_shape = input.shape[:-1] + (self.outfeatures,)
        input = input.reshape(-1, input.shape[-1])

        if self.is_cuda is True and (
                self.kernel_switch_threshold is False or input.shape[0] < self.kernel_switch_threshold):
            out = torch.zeros((input.shape[0], self.outfeatures), device='cuda', dtype=torch.float32)
            if self.bits == 2:
                quant_cuda.vecquant2matmul(input.float(), self.qweight, out, self.scales.float(), self.qzeros,
                                           self.g_idx)
            elif self.bits == 3:
                quant_cuda.vecquant3matmul(input.float(), self.qweight, out, self.scales.float(), self.qzeros,
                                           self.g_idx)
            elif self.bits == 4:
                quant_cuda.vecquant4matmul(input.float(), self.qweight, out, self.scales.float(), self.qzeros,
                                           self.g_idx)
            elif self.bits == 8:
                quant_cuda.vecquant8matmul(input.float(), self.qweight, out, self.scales.float(), self.qzeros,
                                           self.g_idx)
            out = out.half()
        else:
            out = self._unpack_and_scale(input)

        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out

        return out

    def _prepare_intweight(self, linear, scale_zeros):
        intweight = []
        for idx in range(self.infeatures):
            intweight.append(torch.round(
                (linear.weight.data[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(
                torch.int)[:, None])
        return torch.cat(intweight, dim=1)

    def _pack_qweight(self, intweight):
        return super().pack_qweight_blocks(intweight, block_size=32)

    def _pack_qzeros(self, zeros):
        return super().pack_qzeros_blocks(zeros, block_size=32)


    def _unpack_and_scale(self, input):
        if self.bits in [2, 4, 8]:
            unpacked_weight, zeros = self._unpack_and_scale_2_4_8bits()
        elif self.bits == 3:
            unpacked_weight, zeros = self._unpack_and_scale_3bits()
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        weight = unpacked_weight.reshape(unpacked_weight.shape[0] * unpacked_weight.shape[1], unpacked_weight.shape[2])
        weights = (self.scales[self.g_idx] * (weight - zeros[self.g_idx]))
        unpacked_qoutput = torch.matmul(input.half(), weights)

        return unpacked_qoutput

    def _unpack_and_scale_2_4_8bits(self):
        zeros = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
                                          self.wf.unsqueeze(0)).to(torch.int16 if self.bits == 8 else torch.int8)
        torch.bitwise_and(zeros, (2 ** self.bits) - 1, out=zeros)

        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                                           self.wf.unsqueeze(-1)).to(torch.int16 if self.bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2 ** self.bits) - 1, out=weight)

        return weight, zeros

    def _unpack_and_scale_3bits(self):
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = (zeros >> self.wf.unsqueeze(0))
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat([zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]], dim=2)

        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)

        return weight, zeros
