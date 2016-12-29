#include "Graph.h"

int main() {
    Graph g;
    int group_id = g.add_group();
    int batch_size(64), channels(3), data_height(224), data_width(224);
    auto data = std::make_shared<DataOp>(batch_size,
                                         channels,
                                         data_height,
                                         data_width);
    g.add_op("input", data, group_id);
    g.display_ops();
    g.build_forward({"data"});
    return 0;
}
