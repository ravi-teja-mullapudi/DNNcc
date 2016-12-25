#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include "ModelIO.h"
#include "Op.h"

// Enumeration of possible implementations of each op node.
enum OpImpl { REF, HALIDE, CUDNN };

// Enumeration of target architecture for each op implementation.
// Currently supports coarse level granularity of CPU/GPU specific
// CPU and GPU arch support to be added in later.
enum TargetArch { CPU, GPU };

class Graph {
    public:
    std::map<std::string, std::shared_ptr<Op>> ops;
    std::vector<std::map<std::string, std::shared_ptr<Op>>> pipelines;

    Graph() {}

    // Initialize the parameters of operations in the graph using the
    // values from params. The params are matched to ops by name and
    // the order in the op's param list.
    void initialize_params(Params& params);

    // Extract parameters from all the op's in the graph. The params are
    // stored by op name and the order in the op's param list.
    void extract_params(Params& params);

    int add_pipeline();
    int num_pipelines();

    void add_op(std::string name, std::shared_ptr<Op> op, int pipeline_id);

    void build_forward_graph(int pipeline_id, OpImpl impl, TargetArch arch);

    void display_ops();
};
