from abc import ABC, abstractmethod, ABCMeta

import numpy as np
import torch
from torch import nn


class QAbstractLinearLayer(nn.Module, metaclass=ABCMeta):
    def pack(self, linear, scales, zeros):
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone()

        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = self._prepare_intweight(linear, scale_zeros)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        self.qweight = torch.from_numpy(self._pack_qweight(intweight))

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        self.qzeros = torch.from_numpy(self._pack_qzeros(zeros))

    def pack_qweight_blocks(self, intweight, block_size, alpha=1):
        qweight = np.zeros(
            (intweight.shape[0] // block_size * (self.bits * alpha), intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                qweight = self._pack_qweight_3bits(intweight, qweight, i, row)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        return qweight.astype(np.int32)

    def pack_qzeros_blocks(self, zeros, block_size, alpha=1):
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // block_size * (self.bits * alpha)), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            elif self.bits == 3:
                qzeros = self._pack_qzeros_3bits(zeros, qzeros, i, col)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        return qzeros.astype(np.int32)

    def _pack_qweight_3bits(self, intweight, qweight, i, row):
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i))

        i += 10
        qweight[row] |= intweight[i] << 30
        row += 1
        qweight[row] |= (intweight[i] >> 2) & 1
        i += 1
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i) + 1)
        i += 10
        qweight[row] |= intweight[i] << 31
        row += 1
        qweight[row] |= (intweight[i] >> 1) & 0x3
        i += 1
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i) + 2)
        return qweight

    def _pack_qzeros_3bits(self, zeros, qzeros, i, col):
        for j in range(i, i + 10):
            qzeros[:, col] |= zeros[:, j] << (3 * (j - i))

        i += 10
        qzeros[:, col] |= zeros[:, i] << 30
        col += 1
        qzeros[:, col] |= (zeros[:, i] >> 2) & 1
        i += 1
        for j in range(i, i + 10):
            qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
        i += 10
        qzeros[:, col] |= zeros[:, i] << 31
        col += 1
        qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
        i += 1
        for j in range(i, i + 10):
            qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
        return qzeros