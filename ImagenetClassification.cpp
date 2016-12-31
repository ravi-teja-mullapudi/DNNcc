#include "networks/Vgg.h"
#include "Graph.h"

int main() {
    Graph g;
    Vgg16(g);
    g.display_ops();

    g.group_impl[0] = std::make_tuple(OpImpl::HALIDE, TargetArch::CPU);
    g.build_forward({"prob"});
    return 0;
}
