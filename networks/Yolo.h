#include "Graph.h"
std::shared_ptr<Op>
yolo_block(Graph& g, int group_id, std::shared_ptr<Op> in,
           int level, int num_filters, bool has_pool) {

    int filter_height(3), filter_width(3), stride(1);

    auto conv = std::make_shared<Conv2dOp>(num_filters, filter_height,
                                           filter_width, stride, stride, in);

    g.add_op("conv" + std::to_string(level), conv, group_id);

    float slope = 0.1f;
    auto prelu = std::make_shared<ReLUOp>(slope, conv);
    g.add_op("relu" + std::to_string(level), prelu, group_id);

    std::shared_ptr<Op> block_end = prelu;
    if (has_pool) {
        int p_w(2), p_h(2), p_stride(2);
        auto pool = std::make_shared<Pool2dOp>(p_h, p_w, p_stride, p_stride,
                                               PoolType::MAX, prelu);
        g.add_op("pool" + std::to_string(level), pool, group_id);
        block_end = pool;
    }

    return block_end;
}

void yolo_tiny(Graph& g, int batch_size, int channels, int data_height, int data_width) {

    int group_id = g.add_group();

    auto data_sizes = {batch_size, channels, data_height, data_width};
    auto data = std::make_shared<DataOp>(data_sizes);
    g.add_op("data", data, group_id);

    std::vector<int> filter_sizes = {16, 32, 64, 128, 256, 512, 1024};
    std::vector<int> has_pool = {true, true, true, true, false, false, false};
    std::shared_ptr<Op> block_in = data;
    std::shared_ptr<Op> block_out;

    for (size_t i = 0; i < filter_sizes.size(); i++) {
        auto block_out = yolo_block(g, group_id, block_in, i + 1,
                                    filter_sizes[i], has_pool[i]);
        block_in = block_out;
    }

    group_id = g.add_group();

    auto flatten = std::make_shared<FlattenOp>(block_out);
    g.add_op("flatten", flatten, group_id);

    auto fc10 = std::make_shared<AffineOp>(256, flatten);
    g.add_op("fc10", fc10, group_id);

    auto fc11 = std::make_shared<AffineOp>(4096, flatten);
    g.add_op("fc11", fc11, group_id);

    auto relu11 = std::make_shared<ReLUOp>(0.1f, fc11);
    g.add_op("relu11", relu11, group_id);

    auto fc12 = std::make_shared<AffineOp>(4096, relu11);
    g.add_op("fc11", fc12, group_id);
}
