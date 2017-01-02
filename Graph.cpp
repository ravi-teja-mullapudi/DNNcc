#include "Graph.h"

void Graph::set_params(Params& params) {
    for (size_t i = 0; i < groups.size(); i++) {
        for (auto &op: groups[i]) {
            if (op.second->params.size() > 0) {
                assert(params[op.first].size() == op.second->params.size());
                for (size_t p = 0; p < params[op.first].size(); p++) {
                    op.second->params[p] = params[op.first][p];
                    OpImpl impl = std::get<0>(group_impl[i]);
                    if (impl == OpImpl::HALIDE) {
                        Buffer<> buf =
                            get_halide_buffer(params[op.first][p],
                                              op.second->type);
                        halide_ops[op.first]->params[p].set(buf);
                    }
                }
            }
        }
    }
}

void Graph::get_params(Params &params) {
    // Not implemented yet
    assert(0);
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
    assert(group_id == (int)groups.size() - 1);
    assert(ops.find(name) == ops.end());
    ops[name] = op;
    for (size_t g = 0; g < groups.size(); g++) {
        assert(groups[g].find(name) == groups[g].end());
    }
    groups[group_id][name] = op;
    assert(op_name_map.find(op) == op_name_map.end());
    op_name_map[op] = name;
}

void Graph::build_forward_halide(unsigned int group_id) {

    halide_op_ins[group_id] = std::map<std::string, ImageParam>();
    TargetArch arch = std::get<1>(group_impl[group_id]);

    for (auto &in: group_ins[group_id]) {
        assert(ops.at(in)->num_dims() <= 4);
        halide_op_ins[group_id][in] =
            ImageParam(Float(32), ops.at(in)->num_dims());
    }

    for (auto &op_name: order[group_id]) {
        auto op = groups[group_id].at(op_name);
        std::vector<Func> ins;
        for (size_t in = 0; in < op->input_ops.size(); in++) {
            auto in_name = op_name_map[op->input_ops[in]];

            // First check in the inputs to the group
            if (halide_op_ins[group_id].find(in_name) !=
                    halide_op_ins[group_id].end()) {

                ins.push_back(halide_op_ins[group_id][in_name]);
            // Check for inputs within the group
            } else if (groups[group_id].find(in_name) !=
                        groups[group_id].end()) {
                ins.push_back(halide_ops.at(in_name)->output);
            } else {
                assert(0);
            }
        }

        halide_ops[op_name] = std::make_shared<OpHalideImpl>();

        if (std::dynamic_pointer_cast<SumOp>(op) != nullptr) {

            auto op_cast = std::dynamic_pointer_cast<SumOp>(op);
            sum_forward_halide(op_name, op_cast, ins,
                               halide_ops[op_name], arch);

        } else if (std::dynamic_pointer_cast<AffineOp>(op) != nullptr) {

            assert(ins.size() == 1);
            auto op_cast = std::dynamic_pointer_cast<AffineOp>(op);
            affine_forward_halide(op_name, op_cast, ins[0],
                                  halide_ops[op_name], arch);

        } else if (std::dynamic_pointer_cast<Conv2dOp>(op) != nullptr) {

            assert(ins.size() == 1);
            auto op_cast = std::dynamic_pointer_cast<Conv2dOp>(op);
            conv2d_forward_halide(op_name, op_cast, ins[0],
                                  halide_ops[op_name], arch);

        } else if (std::dynamic_pointer_cast<Pool2dOp>(op) != nullptr) {

            assert(ins.size() == 1);
            auto op_cast = std::dynamic_pointer_cast<Pool2dOp>(op);
            pool2d_forward_halide(op_name, op_cast, ins[0],
                                  halide_ops[op_name], arch);

        } else if (std::dynamic_pointer_cast<ReLUOp>(op) != nullptr) {

            assert(ins.size() == 1);
            auto op_cast = std::dynamic_pointer_cast<ReLUOp>(op);
            relu_forward_halide(op_name, op_cast, ins[0],
                                halide_ops[op_name], arch);

        } else if (std::dynamic_pointer_cast<SoftMaxOp>(op) != nullptr) {

            assert(ins.size() == 1);
            auto op_cast = std::dynamic_pointer_cast<SoftMaxOp>(op);
            softmax_forward_halide(op_name, op_cast, ins[0],
                                   halide_ops[op_name], arch);

        } else if (std::dynamic_pointer_cast<LRNOp>(op) != nullptr) {

            assert(ins.size() == 1);
            auto op_cast = std::dynamic_pointer_cast<LRNOp>(op);
            lrn_forward_halide(op_name, op_cast, ins[0],
                               halide_ops[op_name], arch);

        } else if (std::dynamic_pointer_cast<ConcatOp>(op) != nullptr) {

            auto op_cast = std::dynamic_pointer_cast<ConcatOp>(op);
            concat_forward_halide(op_name, op_cast, ins,
                                  halide_ops[op_name], arch);

        } else if (std::dynamic_pointer_cast<FlattenOp>(op) != nullptr) {

            assert(ins.size() == 1);
            auto op_cast = std::dynamic_pointer_cast<FlattenOp>(op);
            flatten_forward_halide(op_name, op_cast, ins[0],
                                   halide_ops[op_name], arch);

        } else if (std::dynamic_pointer_cast<DataOp>(op) != nullptr) {

            assert(ins.size() == 0);
            halide_op_ins[group_id][op_name] =
                ImageParam(Float(32), groups[group_id][op_name]->num_dims());
            auto op_cast = std::dynamic_pointer_cast<DataOp>(op);
            data_forward_halide(op_name, op_cast, halide_op_ins[group_id][op_name],
                                halide_ops[op_name], arch);

        } else {
            std::cerr << "Unknown op" << std::endl;
            assert(0);
        }
    }

    std::vector<Func> outs;

    for (auto &out_name: group_outs[group_id]) {
        auto op = groups[group_id][out_name];
        std::vector<int> buf_sizes;
        for (int d = 0; d < op->num_dims(); d++) {
            buf_sizes.push_back(op->out_size(d));
        }
        op_outs[out_name] = get_ndarray_t(buf_sizes, op->type);
    }

    for (auto &out_name: group_outs[group_id]) {
        auto op = groups[group_id][out_name];
        assert(op->num_dims() <= 4);
        // TODO: Not very certain about what happens when you copy Halide
        // Buffers around. Need to test this.
        halide_op_outs[group_id].
            push_back(get_halide_buffer(op_outs.at(out_name),
                                        op->type));
        outs.push_back(halide_ops[out_name]->output);
    }

    Target target = get_target_from_environment();
    if (arch == TargetArch::GPU) {
        target.set_feature(Target::CUDA);
        target.set_feature(Target::CUDACapability50);
    }

    Pipeline p(outs);
    halide_pipelines[group_id] = p;

    auto start = std::chrono::steady_clock::now();
    halide_pipelines[group_id].compile_jit(target);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Group " << group_id << " compile time: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << "ms" << std::endl;
}

void Graph::build_forward_ref(unsigned int group_id) {
    for (auto &op: groups[group_id]) {
        std::vector<int> buf_sizes;
        for (int d = 0; d < op.second->num_dims(); d++) {
            buf_sizes.push_back(op.second->out_size(d));
        }
        op_outs[op.first] = get_ndarray_t(buf_sizes, op.second->type);
    }
}

void Graph::build_forward_group(unsigned int group_id,
                                const std::vector<std::string>& output_ops) {

    OpImpl impl = std::get<0>(group_impl[group_id]);
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
    std::set<std::string> in_set;
    for (auto &dep: num_prods) {
        if (dep.second == 0) {
            auto op = ops[dep.first];
            if (std::dynamic_pointer_cast<DataOp>(op) != nullptr) {
                if (in_set.find(dep.first) == in_set.end()) {
                    in_set.insert(dep.first);
                }
            } else {
                for (size_t i = 0; i < op->input_ops.size(); i++) {
                    auto in_name = op_name_map.at(op->input_ops[i]);
                    if (in_set.find(in_name) == in_set.end()) {
                        in_set.insert(in_name);
                    }
                }
            }
        }
    }

    for (auto &in: in_set) {
        group_ins[group_id].push_back(in);
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
            group_outs[group_id].push_back(op.first);
        }
    }

    for (auto &op: output_ops) {
        if (groups[group_id].find(op) != groups[group_id].end()) {
            group_outs[group_id].push_back(op);
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
        order[group_id].push_back(curr_op);

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
        build_forward_ref(group_id);
    } else if (impl == OpImpl::HALIDE) {
        build_forward_halide(group_id);
    } else {
        std::cerr << "Unknown implementation" << std::endl;
        assert(0);
    }
}

void Graph::check() {
    // TODO: check if the groups form a graph that makes sense
}

void Graph::build_forward(const std::vector<std::string>& output_ops) {

    for (auto &op: output_ops) {
        assert(ops.find(op) != ops.end());
        graph_outs.push_back(op);
    }

    for (size_t g = 0; g < groups.size(); g++) {
        build_forward_group(g, output_ops);
    }
}

void Graph::display_ops() {
    for (size_t i = 0; i < groups.size(); i++) {
        for (auto &op: groups[i]) {
            std::cout << op.first << std::endl;
        }
    }
}

void Graph::set_halide_group_inputs(unsigned int group_id,
                                    std::map<std::string, NDArray_t>& inputs) {
    // TODO: Handle GPU -> CPU transfers when needed
   for (auto &in: halide_op_ins[group_id]) {
       auto op_in = ops.at(in.first);
       Buffer<> buf;
       if (op_outs.find(in.first) != op_outs.end()) {
           buf = get_halide_buffer(op_outs.at(in.first), op_in->type);
       } else {
           buf = get_halide_buffer(inputs.at(in.first), op_in->type);
       }
       in.second.set(buf);
   }
}

std::map<std::string, NDArray_t>
Graph::run(std::map<std::string, NDArray_t>& inputs) {
    // Run each group in the graph
    for (size_t g = 0; g < groups.size(); g++) {
        OpImpl impl = std::get<0>(group_impl[g]);
        if (impl == OpImpl::HALIDE) {
            // Set the Halide input buffers from corresponding NDArray buffers
            set_halide_group_inputs(g, inputs);
            halide_pipelines[g].
                realize(Realization(halide_op_outs.at(g)));

        } else if (impl == OpImpl::REF) {
            // TODO: Get rid of the giant switch case
            for (auto &op_name: order[g]) {
                auto op = groups[g][op_name];
                if (std::dynamic_pointer_cast<SumOp>(op) != nullptr) {

                    auto op_cast = std::dynamic_pointer_cast<SumOp>(op);
                    std::vector<NDArray<float>> op_ins;
                    for (size_t in = 0; in < op->input_ops.size(); in++) {
                        auto in_op_name = op_name_map[op->input_ops[in]];
                        op_ins.push_back(get_ndarray<float>(op_outs.at(in_op_name)));
                    }

                    NDArray<float>& op_out =
                        get_ndarray<float>(op_outs.at(op_name));

                    sum_forward_ref(op_cast, op_ins, op_out);

                } else if (std::dynamic_pointer_cast<AffineOp>(op) != nullptr) {

                    auto in_op_name = op_name_map[op->input_ops[0]];
                    auto op_cast = std::dynamic_pointer_cast<AffineOp>(op);
                    NDArray<float>& op_in =
                        get_ndarray<float>(op_outs.at(in_op_name));
                    NDArray<float>& op_out =
                        get_ndarray<float>(op_outs.at(op_name));
                    affine_forward_ref(op_cast, op_in, op_out);

                } else if (std::dynamic_pointer_cast<Conv2dOp>(op) != nullptr) {

                    auto in_op_name = op_name_map[op->input_ops[0]];
                    auto op_cast = std::dynamic_pointer_cast<Conv2dOp>(op);
                    NDArray<float>& op_in =
                        get_ndarray<float>(op_outs.at(in_op_name));
                    NDArray<float>& op_out =
                        get_ndarray<float>(op_outs.at(op_name));
                    conv2d_forward_ref(op_cast, op_in, op_out);

                } else if (std::dynamic_pointer_cast<Pool2dOp>(op) != nullptr) {

                    auto in_op_name = op_name_map[op->input_ops[0]];
                    auto op_cast = std::dynamic_pointer_cast<Pool2dOp>(op);
                    NDArray<float>& op_in =
                        get_ndarray<float>(op_outs.at(in_op_name));
                    NDArray<float>& op_out =
                        get_ndarray<float>(op_outs.at(op_name));
                    pool2d_forward_ref(op_cast, op_in, op_out);

                } else if (std::dynamic_pointer_cast<ReLUOp>(op) != nullptr) {

                    auto in_op_name = op_name_map[op->input_ops[0]];
                    auto op_cast = std::dynamic_pointer_cast<ReLUOp>(op);
                    NDArray<float>& op_in =
                        get_ndarray<float>(op_outs.at(in_op_name));
                    NDArray<float>& op_out =
                        get_ndarray<float>(op_outs.at(op_name));
                    relu_forward_ref(op_cast, op_in, op_out);

                } else if (std::dynamic_pointer_cast<SoftMaxOp>(op) != nullptr) {

                    auto in_op_name = op_name_map[op->input_ops[0]];
                    auto op_cast = std::dynamic_pointer_cast<SoftMaxOp>(op);
                    NDArray<float>& op_in =
                        get_ndarray<float>(op_outs.at(in_op_name));
                    NDArray<float>& op_out =
                        get_ndarray<float>(op_outs.at(op_name));
                    softmax_forward_ref(op_cast, op_in, op_out);

                } else if (std::dynamic_pointer_cast<LRNOp>(op) != nullptr) {

                    auto in_op_name = op_name_map[op->input_ops[0]];
                    auto op_cast = std::dynamic_pointer_cast<LRNOp>(op);
                    NDArray<float>& op_in =
                        get_ndarray<float>(op_outs.at(in_op_name));
                    NDArray<float>& op_out =
                        get_ndarray<float>(op_outs.at(op_name));
                    lrn_forward_ref(op_cast, op_in, op_out);

                } else if (std::dynamic_pointer_cast<ConcatOp>(op) != nullptr) {

                    auto op_cast = std::dynamic_pointer_cast<ConcatOp>(op);
                    std::vector<NDArray<float>> op_ins;
                    for (size_t in = 0; in < op->input_ops.size(); in++) {
                        auto in_op_name = op_name_map[op->input_ops[in]];
                        op_ins.push_back(get_ndarray<float>(op_outs.at(in_op_name)));
                    }

                    NDArray<float>& op_out =
                        get_ndarray<float>(op_outs.at(op_name));

                    concat_forward_ref(op_cast, op_ins, op_out);

                } else if (std::dynamic_pointer_cast<FlattenOp>(op) != nullptr) {

                    auto in_op_name = op_name_map[op->input_ops[0]];
                    auto op_cast = std::dynamic_pointer_cast<FlattenOp>(op);
                    NDArray<float>& op_in =
                        get_ndarray<float>(op_outs.at(in_op_name));
                    NDArray<float>& op_out =
                        get_ndarray<float>(op_outs.at(op_name));
                    flatten_forward_ref(op_cast, op_in, op_out);

                } else if (std::dynamic_pointer_cast<DataOp>(op) != nullptr) {

                    auto op_cast = std::dynamic_pointer_cast<DataOp>(op);
                    NDArray<float>& op_in =
                        get_ndarray<float>(inputs.at(op_name));
                    NDArray<float>& op_out =
                        get_ndarray<float>(op_outs.at(op_name));
                    data_forward_ref(op_cast, op_in, op_out);

                } else {
                    std::cerr << "Unknown op" << std::endl;
                    assert(0);
                }
            }
        } else {
            std::cerr << "Unknown implementation" << std::endl;
            assert(0);
        }
    }

    std::map<std::string, NDArray_t> outputs;
    for (auto &op: graph_outs) {
        outputs[op] = op_outs.at(op);
    }

    return outputs;
}
