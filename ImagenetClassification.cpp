#include "networks/Vgg.h"
#include "networks/Googlenet.h"
#include "Graph.h"

int main() {
    Graph vgg;
    Vgg16(vgg, 16, 3, 224, 224);
    vgg.display_ops();

    vgg.group_impl[0] = std::make_tuple(OpImpl::HALIDE, TargetArch::CPU);
    vgg.build_forward({"prob"});

    Graph googlenet;
    Googlenet(googlenet, 16, 3, 224, 224);

    googlenet.display_ops();
    for (size_t i = 0; i < googlenet.groups.size(); i++) {
        googlenet.group_impl[i] = std::make_tuple(OpImpl::HALIDE, TargetArch::CPU);
    }

    googlenet.build_forward({"prob"});
    return 0;
}
