#include "Graph.h"

void Graph::initialize_params(Params &params) {
    for (size_t i = 0; i < groups.size(); i++) {
        for (auto &op: groups[i]) {
            if (op.second->params.size() > 0) {
                assert(params[op.first].size() == op.second->params.size());
                for (size_t p = 0; p < params[op.first].size(); p++) {
                    op.second->params[p] = params[op.first][p];
                }
            }
        }
    }
}

int Graph::add_group() {
    groups.push_back(std::map<std::string, std::shared_ptr<Op>>());
    int group_id = groups.size() - 1;
    // Default implementation is the reference one.
    group_impl[group_id] = std::make_tuple(OpImpl::REF, TargetArch::CPU);
    return groups.size() - 1;
}

int Graph::num_groups() {
    return groups.size();
}

void Graph::add_op(std::string name, std::shared_ptr<Op> op, int group_id) {
    assert(group_id < (int) groups.size());
    assert(groups[group_id].find(name) == groups[group_id].end());
    groups[group_id][name] = op;
}

void Graph::build_forward_halide(unsigned int group_id,
                                 const std::vector<std::string>& order,
                                 const std::vector<std::string>& group_ins,
                                 const std::vector<std::string>& group_outs) {

    holide_op_ins[group_id] = std::map<std::string, ImageParam>();
    for (auto &in: group_ins) {
        assert(groups[group_id][in]->num_dims() <= 4);
        holide_op_ins[group_id][in] =
            ImageParam(Float(32), groups[group_id][in]->num_dims());
    }

    for (auto &op_name: order) {
        auto op = groups[group_id][op_name];
        if (std::dynamic_pointer_cast<AffineOp>(op) != nullptr) {
        } else if (std::dynamic_pointer_cast<Conv2dOp>(op) != nullptr) {
        } else if (std::dynamic_pointer_cast<Pool2dOp>(op) != nullptr) {
        } else if (std::dynamic_pointer_cast<ReLUOp>(op) != nullptr) {
        } else if (std::dynamic_pointer_cast<SoftMaxOp>(op) != nullptr) {
        } else if (std::dynamic_pointer_cast<LRNOp>(op) != nullptr) {
        } else if (std::dynamic_pointer_cast<ConcatOp>(op) != nullptr) {
        } else if (std::dynamic_pointer_cast<FlattenOp>(op) != nullptr) {
        } else if (std::dynamic_pointer_cast<DataOp>(op) != nullptr) {
        }
    }
}

void Graph::build_forward_ref(unsigned int group_id,
                              const std::vector<std::string>& order,
                              const std::vector<std::string>& group_ins,
                              const std::vector<std::string>& group_outs) {
}

void Graph::build_forward_group(unsigned int group_id,
                                std::vector<std::string>& output_ops) {
    OpImpl impl = std::get<0>(group_impl[group_id]);
    std::vector<std::string> order, group_ins, group_outs;
    std::map<std::string, int> num_prods;

    // Find a valid execution order for the ops.
    for (auto &op: groups[group_id]) {
        assert(num_prods.find(op.first) == num_prods.end());
        num_prods[op.first] = 0;
        // Count number of dependecies in the group.
        for (auto &in_op: op.second->input_ops) {
            bool found = false;
            for (auto &s: groups[group_id]) {
               if (s.second == in_op) {
                    found = true;
                    break;
                }
            }
            if (found) {
                num_prods[op.first] += 1;
            }
        }
    }

    // Get the input ops for the group.
    for (auto &dep: num_prods) {
        if (dep.second == 0) {
            group_ins.push_back(dep.first);
        }
    }

    // Get the output ops for the group.
    for (auto &op: groups[group_id]) {
        bool used_outside_group = false;
        for (size_t g = 0; g < groups.size(); g++) {
            if (g != group_id && !used_outside_group) {
                for (auto &s: groups[g]) {
                    for (auto &in: s.second->input_ops) {
                        if (in == op.second) {
                            used_outside_group = true;
                            break;
                        }
                    }
                }
            }
        }

        if (used_outside_group) {
            group_outs.push_back(op.first);
        }
    }

    for (auto &op: output_ops) {
        if (groups[group_id].find(op) != groups[group_id].end()) {
            group_outs.push_back(op);
        }
    }

    while (num_prods.size() > 0) {
        std::string curr_op;
        for (auto &op: num_prods) {
            if (op.second == 0) {
                curr_op = op.first;
            }
        }

        num_prods.erase(curr_op);
        order.push_back(curr_op);

        for (auto &op: groups[group_id]) {
            for (auto &in_op: op.second->input_ops) {
                if (groups[group_id][curr_op] == in_op) {
                    num_prods[op.first] -= 1;
                }
            }
        }
    }

    if (impl == OpImpl::REF) {
        // Create input and output buffers for each op.
        build_forward_ref(group_id, order, group_ins, group_outs);
    } else if (impl == OpImpl::HALIDE) {
        build_forward_halide(group_id, order, group_ins, group_outs);
    } else {
        // TODO: Other implementations.
        assert(0);
    }
}

void Graph::check() {
    // TODO: check if the groups form a graph that makes sense
}

void Graph::build_forward(std::vector<std::string>& output_ops) {
}

void Graph::display_ops() {
    for (size_t i = 0; i < groups.size(); i++) {
        for (auto &op: groups[i]) {
            std::cout << op.first << std::endl;
        }
    }
}
