#pragma once

#include <vector>
#include <tuple>
#include <memory>
#include <iostream>
#include "ModelIO.h"
#include "Op.h"
#include "OpRef.h"
#include "OpImpl.h"
#include "OpHalide.h"

class Graph {
    public:
    // TODO: consolidate into a class
    std::vector<std::map<std::string, std::shared_ptr<Op>>> groups;
    std::map<std::shared_ptr<Op>, std::string> op_name_map;
    std::map<int, std::tuple<OpImpl, TargetArch>> group_impl;

    // TODO: consolidate into a class
    std::map<std::string, std::shared_ptr<OpHalideImpl>> halide_ops;
    std::map<int, Pipeline> halide_pipelines;
    std::map<int, std::map<std::string, ImageParam>> halide_op_ins;

    std::map<std::string, Buffer<>> halide_op_outs;
    std::map<std::string, NDArray_t> op_outs;

    std::map<int, std::vector<std::string>> order;
    std::map<int, std::vector<std::string>> group_ins;
    std::map<int, std::vector<std::string>> group_outs;

    Graph() {}

    // Initialize the parameters of operations in the graph using the
    // values from params. The params are matched to ops by name and
    // the order in the op's param list.
    void initialize_params(Params& params);

    // Extract parameters from all the op's in the graph. The params are
    // stored by op name and the order in the op's param list.
    void extract_params(Params& params);

    int add_group();
    int num_groups();

    void add_op(std::string name, std::shared_ptr<Op> op, int group_id);

    void check();

    void build_forward_halide(unsigned int group_id);

    void build_forward_ref(unsigned int group_id);

    void build_forward_group(unsigned int group_id,
                             const std::vector<std::string>& output_ops);

    void build_forward(const std::vector<std::string>& output_ops);

    std::map<std::string, NDArray_t>
        run(std::map<std::string, NDArray_t>& inputs);

    void display_ops();
};
