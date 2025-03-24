# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class aten_extend_base(torch.nn.Module):
    def forward(self, x):
        list1 = self.prepare_input(x)
        list2 = self.prepare_input(x)
        list1.extend(list2)
        return torch.cat(list1, dim=0)

    def prepare_input(self, x):
        return [x, x * 2]


class aten_extend_empty(torch.nn.Module):
    def forward(self, x):
        list1 = []
        list2 = [x, x * 2]
        list1.extend(list2)
        return torch.cat(list1, dim=0)


class TestExtend(PytorchLayerTest):
    def _prepare_input(self):
        data = np.random.randn(2, 1, 3)
        return (data,)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_extend(self, ie_device, precision, ir_version):
        self._test(aten_extend_base(), None, ["aten::extend", "aten::cat", "prim::ListConstruct"],
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

    @pytest.mark.parametrize(("in_types"), [
        # Two inputs - basic type combinations
        (np.float32, np.int32),
        (np.int32, np.float32),
        (np.float16, np.float32),
    ])
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_align_types_extend(self, ie_device, precision, ir_version, in_types, trace_model):
        class aten_align_types_extend(torch.nn.Module):
            def forward(self, x, y):
                list1 = [x, x * 2]
                list2 = [y, y * 2]
                list1.extend(list2)
                return torch.cat(list1, dim=0)

        self._test(aten_align_types_extend(), None, ["aten::extend", "aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"in_types": in_types}, trace_model=trace_model)


class TestExtendAlignTypesPT(PytorchLayerTest):
    def _prepare_input(self, in_types):
        in_vals = [np.random.randn(2, 2, 3).astype(in_types[0])]
        return in_vals

    @pytest.mark.parametrize(("in_types"), [
        # Two inputs (param, const) - basic type combinations
        (np.float32, torch.int32),
        (np.int32, torch.float32),
        (np.float16, torch.float32),
    ])
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_align_types_extend(self, ie_device, precision, ir_version, in_types, trace_model):
        class aten_align_types_extend(torch.nn.Module):
            def __init__(self):
                super(aten_align_types_extend, self).__init__()
                self.y = torch.randn(2, 1, 3).to(in_types[1])

            def forward(self, x):
                x_ = torch.split(x, 1, 1)[1]
                list1 = [x_, x_ * 2]
                list2 = [self.y, self.y * 2]
                list1.extend(list2)
                return torch.cat(list1, dim=0)

        self._test(aten_align_types_extend(), None, ["aten::extend", "aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"in_types": in_types}, trace_model=trace_model) 