#include "Graph.h"

void Graph::initialize_params(Params &params) {
    for (size_t i = 0; i < pipelines.size(); i++) {
        for (auto &op: pipelines[i]) {
            if (op.second->params.size() > 0) {
                assert(params[op.first].size() == op.second->params.size());
                for (size_t p = 0; p < params[op.first].size(); p++) {
                    op.second->params[p] = params[op.first][p];
                }
            }
        }
    }
}

int Graph::add_pipeline() {
    pipelines.push_back(std::map<std::string, std::shared_ptr<Op>>());
    return pipelines.size() - 1;
}

int Graph::num_pipelines() {
    return pipelines.size();
}

void Graph::add_op(std::string name, std::shared_ptr<Op> op, int pipeline_id) {
    assert(pipeline_id < (int) pipelines.size());
    assert(pipelines[pipeline_id].find(name) == pipelines[pipeline_id].end());
    pipelines[pipeline_id][name] = op;
}

void build_forward_graph(int pipeline_id, OpImpl impl, TargetArch arch) {
    // Get the inputs and output shapes for the pipeline.
    if (impl == OpImpl::REF) {
        // Create input and output buffers for each op.

        // Find a valid execution order for the ops.

    } else if (impl == OpImpl::HALIDE) {

    } else {
        // TODO: Other implementations.
        assert(0);
    }
}

void Graph::display_ops() {
    for (size_t i = 0; i < pipelines.size(); i++) {
        for (auto &op: pipelines[i]) {
            std::cout << op.first << std::endl;
        }
    }
}
