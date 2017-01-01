#include "Graph.h"
#include "Utils.h"

void test_data() {
    Graph g;
    int group_id = g.add_group();
    int batch_size(64), channels(3), data_height(224), data_width(224);
    auto data_sizes = {batch_size, channels, data_height, data_width};
    auto data = std::make_shared<DataOp>(data_sizes);

    g.add_op("data", data, group_id);

    g.group_impl[group_id] = std::make_tuple(OpImpl::HALIDE, TargetArch::CPU);

    g.build_forward({"data"});

    g.display_ops();
}

void test_conv2d() {

    Graph g_h;
    int group_id = g_h.add_group();
    auto data_sizes = {16, 3, 224, 224};

    auto data = std::make_shared<DataOp>(data_sizes);
    g_h.add_op("data", data, group_id);

    auto conv = std::make_shared<Conv2dOp>(64, 3, 3, 1, 1, data);
    g_h.add_op("conv", conv, group_id);

    g_h.group_impl[group_id] = std::make_tuple(OpImpl::HALIDE, TargetArch::CPU);

    g_h.build_forward({"conv"});

    Graph g_ref;
    group_id = g_ref.add_group();
    g_ref.add_op("data", data, group_id);
    g_ref.add_op("conv", conv, group_id);
    g_ref.build_forward({"conv"});

    GaussianGenerator<float> rgen(1.0f, 0.1f);

    Params params;
    NDArray<float> W({64, 3, 3, 3});
    W.initialize(rgen);
    NDArray<float> b({64});
    b.initialize(rgen);

    params["conv"].push_back(W);
    params["conv"].push_back(b);

    g_h.set_params(params);
    g_ref.set_params(params);

    NDArray<float> d({16, 3, 224, 224});
    d.initialize(rgen);

    std::map<std::string, NDArray_t> ins;
    ins["data"] = d;

    //auto start = std::chrono::steady_clock::now();
    std::map<std::string, NDArray_t> outs_h = g_h.run(ins);
    //auto end = std::chrono::steady_clock::now();
    //std::cout << "Runtime: " <<
    //    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //    << "ms" << std::endl;

    NDArray<float> out_h = get_ndarray<float>(outs_h["conv"]);

    //start = std::chrono::steady_clock::now();
    std::map<std::string, NDArray_t> outs_ref = g_ref.run(ins);
    //end = std::chrono::steady_clock::now();
    //std::cout << "Runtime: " <<
    //    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //    << "ms" << std::endl;

    NDArray<float> out_ref = get_ndarray<float>(outs_ref["conv"]);

    assert(is_nearly_equal(out_ref(0, 1, 1, 1),
                           out_h(0, 1, 1, 1)));
}

void test_sum() {

    Graph g;
    int group_id = g.add_group();
    int extent(1024);

    NDArray<float> a1({extent});
    NDArray<float> a2({extent});

    a1.initialize(1.0f);
    a2.initialize(1.0f);

    auto data_sizes = {extent};
    auto data1 = std::make_shared<DataOp>(data_sizes);
    g.add_op("data1", data1, group_id);

    auto data2 = std::make_shared<DataOp>(data_sizes);
    g.add_op("data2", data2, group_id);

    std::vector<std::shared_ptr<Op>> sum_ins = {data1, data2};
    auto sum = std::make_shared<SumOp>(data_sizes, sum_ins);
    g.add_op("sum", sum, group_id);

    g.group_impl[group_id] = std::make_tuple(OpImpl::HALIDE, TargetArch::CPU);

    g.build_forward({"sum"});

    g.display_ops();

    std::map<std::string, NDArray_t> ins;
    ins["data1"] = a1;
    ins["data2"] = a2;
    std::map<std::string, NDArray_t> outs = g.run(ins);

    NDArray<float> sum_out = get_ndarray<float>(outs["sum"]);
    for (int i = 0; i < extent; i++) {
       assert(is_nearly_equal(sum_out(i), 2.0f));
    }
}

int main() {
    test_data();
    test_sum();
    test_conv2d();
    return 0;
}
