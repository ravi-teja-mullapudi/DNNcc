#include "Graph.h"

void Graph::initialize_params(Params &params) {
    for (auto &op: ops) {
        if (op.second->params.size() > 0) {
            assert(params[op.first].size() == op.second->params.size());
            for (size_t p = 0; p < params[op.first].size(); p++) {
                op.second->params[p] = params[op.first][p];
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
    assert(ops.find(name) == ops.end());
    assert(pipeline_id < (int) pipelines.size());
    ops[name] = op;
    assert(pipelines[pipeline_id].find(name) == pipelines[pipeline_id].end());
    pipelines[pipeline_id][name] = op;
}

void Graph::display_ops() {
    for (auto &op: ops) {
        std::cout << op.first << std::endl;
    }
}
