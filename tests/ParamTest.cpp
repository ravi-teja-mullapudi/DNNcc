#include "networks/Vgg.h"
#include "Graph.h"

int main() {
    Graph g;
    Vgg16(g);

    g.group_impl[0] = std::make_tuple(OpImpl::HALIDE, TargetArch::CPU);
    g.build_forward({"prob"});
    std::cout << "Graph compiled" << std::endl;

    Params params;
    load_model_from_disk("/home/ravi/DNNcc/params/vgg16.bin", params);
    std::cout << "Model loaded from disk" << std::endl;

    g.set_params(params);

    NDArray<float> d({16, 3, 224, 224});
    d.initialize(0.0f);

    std::map<std::string, NDArray_t> ins;
    ins["data"] = d;

    auto start = std::chrono::steady_clock::now();
    std::map<std::string, NDArray_t> outs = g.run(ins);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Runtime: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << "ms" << std::endl;

    NDArray<float> out = get_ndarray<float>(outs["prob"]);
    return 0;
}
