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
    return 0;
}
