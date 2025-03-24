# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class aten_extend(torch.nn.Module):
    def forward(self, x, y):
        list1 = [x, x * 2]
        list2 = [y, y * 2]
        list1.extend(list2)
        return torch.cat(list1, dim=0)

class aten_extend_empty(torch.nn.Module):
    def forward(self, x, y):
        list1 = []
        list2 = [y, y * 2]
        list1.extend(list2)
        return torch.cat(list1, dim=0)

class aten_extend_mixed_types(torch.nn.Module):
    def forward(self, x, y):
        list1 = [x, x * 2]
        list2 = [y, y * 2]
        list1.extend(list2)
        return torch.cat(list1, dim=0)

class TestExtend(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        x = np.random.randn(2, 1, 3)
        y = np.random.randn(2, 1, 3)
        return (x, y)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_extend(self, ie_device, precision, ir_version):
        self._test(aten_extend(), None, ["aten::extend", "aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_extend_empty(self, ie_device, precision, ir_version):
        self._test(aten_extend_empty(), None, ["aten::extend", "aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version)

class TestExtendAlignTypes(PytorchLayerTest):
    def _prepare_input(self, in_types):
        in_vals = []
        for i in range(len(in_types)):
            dtype = in_types[i]
            in_vals.append(np.random.randn(2, 1, 3).astype(dtype))
        return in_vals

    def create_model(self, in_types):
        class aten_align_types_extend(torch.nn.Module):
            def forward(self, x, y):
                list1 = [x, x * 2]
                list2 = [y, y * 2]
                list1.extend(list2)
                return torch.cat(list1, dim=0)
        return aten_align_types_extend()

    @pytest.mark.parametrize(("in_types"), [
        # Two inputs
        (np.float32, np.int32),
        (np.int32, np.float32),
        (np.float16, np.float32),
        (np.int16, np.float16),
        (np.int32, np.int64),
        (np.float16, np.bfloat16),
        (np.float32, np.bfloat16),
    ])
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_align_types_extend(self, ie_device, precision, ir_version, in_types, trace_model):
        self._test(self.create_model(in_types), None, ["aten::extend", "aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"in_types": in_types}, trace_model=trace_model) 