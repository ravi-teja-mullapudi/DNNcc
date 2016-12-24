#include "Graph.h"

void Graph::initialize_params(Params &params) {
    for (auto &op: ops) {
        if (op.second.params.size() > 0) {
            assert(params[op.first].size() == op.second.params.size());
            for (size_t p = 0; p < params[op.first].size(); p++) {
                op.second.params[p] = params[op.first][p];
            }
        }
    }
}

void Graph::display_ops() {
    for (auto &op: ops) {
        std::cout << op.first << std::endl;
    }
}
