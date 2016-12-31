#include "networks/Vgg.h"
#include "Graph.h"

int main() {
    Graph g;
    Vgg16(g);
    g.display_ops();

    g.group_impl[0] = std::make_tuple(OpImpl::HALIDE, TargetArch::CPU);
    g.build_forward({"prob"});
    std::cout << "Graph compiled" << std::endl;

    Params params;
    load_model_from_disk("/home/ravi/DNNcc/params/vgg16.bin", params);
    std::cout << "Model loaded from disk" << std::endl;

    g.set_params(params);

    return 0;
}
