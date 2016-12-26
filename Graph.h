#pragma once

#include <vector>
#include <tuple>
#include <memory>
#include <iostream>
#include "ModelIO.h"
#include "Op.h"
#include "OpImpl.h"
#include "OpHalide.h"

class Graph {
    public:
    std::vector<std::map<std::string, std::shared_ptr<Op>>> groups;
    std::map<std::shared_ptr<Op>, std::shared_ptr<OpHalideImpl>> halide_ops;
    std::map<int, std::tuple<OpImpl, TargetArch>> group_impl;

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
    void build_forward_group(int group_id, std::vector<std::string>& output_ops);
    void build_forward(std::vector<std::string>& output_ops);

    void display_ops();
};
