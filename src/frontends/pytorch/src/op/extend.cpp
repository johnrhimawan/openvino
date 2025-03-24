// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/convert_promote_types.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_extend(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    
    // Get the list to extend and the list to extend with
    const auto&& target_list = get_list_as_outputs(context.get_input(0));
    const auto&& source_list = get_list_as_outputs(context.get_input(1));
    
    if (target_list.empty() || source_list.empty()) {
        // If either list is empty, create a framework node
        auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), 
            OutputVector{context.get_input(0), context.get_input(1)}, 1);
        return {context.mark_node(fw_node)};
    }

    // Combine the lists
    OutputVector combined_inputs;
    combined_inputs.insert(combined_inputs.end(), target_list.begin(), target_list.end());
    combined_inputs.insert(combined_inputs.end(), source_list.begin(), source_list.end());

    // Handle mixed types if necessary
    const auto first_in_type = combined_inputs.front().get_element_type();
    const bool is_mixed_type = std::any_of(std::next(combined_inputs.begin()),
                                          combined_inputs.end(),
                                          [&first_in_type](const ov::Output<ov::Node>& input) {
                                              return input.get_element_type() != first_in_type ||
                                                     input.get_element_type() == ov::element::dynamic;
                                          });

    if (is_mixed_type) {
        auto node_of_type = combined_inputs[0];
        for (size_t i = 1; i < combined_inputs.size(); ++i) {
            auto cpt = context.mark_node(std::make_shared<v14::ConvertPromoteTypes>(node_of_type, combined_inputs[i], true));
            node_of_type = cpt->output(0);
            combined_inputs[i] = cpt->output(1);
        }

        combined_inputs[0] = node_of_type;
        const auto unified_type = node_of_type.get_element_type();
        for (size_t i = 1; i < combined_inputs.size(); ++i) {
            if (combined_inputs[i].get_element_type() != unified_type ||
                combined_inputs[i].get_element_type() == ov::element::dynamic) {
                combined_inputs[i] = context.mark_node(std::make_shared<v1::ConvertLike>(combined_inputs[i], node_of_type));
            }
        }
    }

    // Concatenate all elements along the first dimension
    auto concat = std::make_shared<v0::Concat>(combined_inputs, 0);
    return {context.mark_node(concat)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov 