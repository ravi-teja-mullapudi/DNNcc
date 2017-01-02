#include "Graph.h"

std::shared_ptr<Op>
inception_module(Graph& g, std::string prefix, std::shared_ptr<Op> in, int group_id,
                 int _1x1_filters, int _3x3_reduce_filters, int _3x3_filters,
                 int _5x5_reduce_filters, int _5x5_filters,
                 int _pool_proj_filters) {

    int num_filters(_1x1_filters), filter_width(1), filter_height(1), stride(1);
    auto _1x1  = std::make_shared<Conv2dOp>(num_filters, filter_height, filter_width,
                                            stride, stride, in);
    g.add_op(prefix + "1x1", _1x1, group_id);

    float slope = 0.0f;
    auto relu_1x1 = std::make_shared<ReLUOp>(slope, _1x1);
    g.add_op(prefix + "relu_1x1", relu_1x1, group_id);

    num_filters = _3x3_reduce_filters;
    auto _3x3_reduce  = std::make_shared<Conv2dOp>(num_filters, filter_height, filter_width,
                                                   stride, stride, relu_1x1);
    g.add_op(prefix + "3x3_reduce", _3x3_reduce, group_id);

    auto relu_3x3_reduce = std::make_shared<ReLUOp>(slope, _3x3_reduce);
    g.add_op(prefix + "relu_3x3_reduce", relu_3x3_reduce, group_id);

    num_filters = _3x3_filters;
    filter_height = 3;
    filter_width = 3;

    auto _3x3  = std::make_shared<Conv2dOp>(num_filters, filter_height, filter_width,
                                            stride, stride, relu_3x3_reduce);
    g.add_op(prefix + "3x3", _3x3, group_id);

    auto relu_3x3 = std::make_shared<ReLUOp>(slope, _3x3);
    g.add_op(prefix + "relu_3x3", relu_3x3, group_id);

    num_filters = _5x5_reduce_filters;
    filter_width = 1;
    filter_height = 1;

    auto _5x5_reduce  = std::make_shared<Conv2dOp>(num_filters, filter_height,
                                                   filter_width, stride, stride, in);
    g.add_op(prefix + "5x5_reduce", _5x5_reduce, group_id);

    auto relu_5x5_reduce = std::make_shared<ReLUOp>(slope, _5x5_reduce);
    g.add_op(prefix + "relu_5x5_reduce", relu_5x5_reduce, group_id);

    num_filters = _5x5_filters;
    filter_width = 5;
    filter_height = 5;

    auto _5x5  = std::make_shared<Conv2dOp>(num_filters, filter_height, filter_width,
                                            stride, stride, relu_5x5_reduce);
    g.add_op(prefix + "5x5", _5x5, group_id);

    auto relu_5x5 = std::make_shared<ReLUOp>(slope, _5x5);
    g.add_op(prefix + "relu_5x5", relu_5x5, group_id);

    int p_w(3), p_h(3), p_stride(1);

    auto pool = std::make_shared<Pool2dOp>(p_h, p_w,  p_stride, p_stride,
                                           PoolType::MAX, in);
    g.add_op(prefix + "pool", pool, group_id);

    num_filters = _pool_proj_filters;
    filter_width = 1;
    filter_height = 1;

    auto pool_proj  = std::make_shared<Conv2dOp>(num_filters, filter_height,
                                                 filter_width, stride, stride, pool);
    g.add_op(prefix + "pool_proj", pool_proj, group_id);

    auto relu_pool_proj = std::make_shared<ReLUOp>(slope, pool_proj);
    g.add_op(prefix + "relu_pool_proj", relu_pool_proj, group_id);

    std::vector<std::shared_ptr<Op>> concat_ins =
                {relu_1x1, relu_3x3, relu_5x5, relu_pool_proj};
    auto output = std::make_shared<ConcatOp>(concat_ins);
    g.add_op(prefix + "output", output, group_id);

    return output;
}

void Googlenet(Graph& g, int batch_size, int channels,
               int data_height, int data_width) {

    int group_id = g.add_group();
    auto data_sizes = {batch_size, channels, data_height, data_width};
    auto data = std::make_shared<DataOp>(data_sizes);
    g.add_op("data", data, group_id);

    int num_filters(64), filter_height(7), filter_width(7), stride(2);
    auto conv1_7x7_s2  = std::make_shared<Conv2dOp>(num_filters, filter_height, filter_width,
                                                    stride, stride, data);
    g.add_op("conv1/7x7_s2", conv1_7x7_s2, group_id);

    float slope = 0.0f;
    auto conv1_relu_7x7 = std::make_shared<ReLUOp>(slope, conv1_7x7_s2);
    g.add_op("conv1/relu_7x7", conv1_relu_7x7, group_id);

    int p_w(3), p_h(3), p_stride(2);

    auto pool1_3x3_s2 = std::make_shared<Pool2dOp>(p_h, p_w, p_stride, p_stride,
                                                   PoolType::MAX, conv1_relu_7x7);
    g.add_op("pool1/3x3_s2", pool1_3x3_s2, group_id);

    int local_size = 5;
    float alpha = 0.0001;
    float beta = 0.75;

    auto pool1_norm1 = std::make_shared<LRNOp>(local_size, alpha, beta, pool1_3x3_s2);

    g.add_op("pool1/norm1", pool1_norm1, group_id);

    num_filters = 64;
    filter_width = 1;
    filter_height = 1;
    stride = 1;
    auto conv2_3x3_reduce  = std::make_shared<Conv2dOp>(num_filters, filter_height,
                                                        filter_width, stride, stride,
                                                        pool1_norm1);
    g.add_op("conv2/3x3_reduce", conv2_3x3_reduce, group_id);

    auto relu_3x3_reduce = std::make_shared<ReLUOp>(slope, conv2_3x3_reduce);
    g.add_op("conv2/relu_3x3_reduce", relu_3x3_reduce, group_id);

    num_filters = 192;
    filter_width = 3;
    filter_height = 3;
    auto _3x3  = std::make_shared<Conv2dOp>(num_filters, filter_height, filter_width,
                                            stride, stride, relu_3x3_reduce);
    g.add_op("conv2/3x3", _3x3, group_id);

    auto relu_3x3 = std::make_shared<ReLUOp>(slope, _3x3);
    g.add_op("conv2/relu_3x3", relu_3x3, group_id);

    auto conv2_norm2 = std::make_shared<LRNOp>(local_size, alpha, beta, relu_3x3);
    g.add_op("conv2/norm2", conv2_norm2, group_id);

    auto pool2_3x3_s2 = std::make_shared<Pool2dOp>(p_h, p_w, p_stride, p_stride,
                                                   PoolType::MAX, conv2_norm2);
    g.add_op("pool2/3x3_s2", pool2_3x3_s2, group_id);

    group_id = g.add_group();

    auto output_3a = inception_module(g, "inception_3a/", pool2_3x3_s2, group_id,
                                      64, 96, 128, 16, 32, 32);
    auto output_3b = inception_module(g, "inception_3b/", output_3a, group_id,
                                      128, 128, 192, 32, 96, 64);

    auto pool3_3x3_s2 = std::make_shared<Pool2dOp>(p_h, p_w, p_stride, p_stride,
                                                   PoolType::MAX, output_3b);
    g.add_op("pool3/3x3_s2", pool3_3x3_s2, group_id);

    group_id = g.add_group();

    auto output_4a = inception_module(g, "inception_4a/", pool3_3x3_s2, group_id,
                                      192, 96, 208, 16, 48, 64);
    auto output_4b = inception_module(g, "inception_4b/", output_4a, group_id,
                                      160, 112, 224, 24, 64, 64);

    group_id = g.add_group();

    auto output_4c = inception_module(g, "inception_4c/", output_4a, group_id,
                                      128, 128, 256, 24, 64, 64);

    auto output_4d = inception_module(g, "inception_4d/", output_4c, group_id,
                                      112, 144, 288, 32, 64, 64);

    group_id = g.add_group();

    auto output_4e = inception_module(g, "inception_4e/", output_4d, group_id,
                                      256, 160, 320, 32, 128, 128);

    auto pool4_3x3_s2 = std::make_shared<Pool2dOp>(p_h, p_w, p_stride, p_stride,
                                                   PoolType::MAX, output_4e);

    g.add_op("pool4/3x3_s2", pool4_3x3_s2, group_id);

    group_id = g.add_group();

    auto output_5a = inception_module(g, "inception_5a/", output_4e, group_id,
                                      256, 160, 320, 32, 128, 128);
    auto output_5b = inception_module(g, "inception_5b/", output_5a, group_id,
                                      384, 192, 384, 48, 128, 128);

    p_w = 7;
    p_h = 7;
    p_stride = 1;
    auto pool5_7x7_s1 = std::make_shared<Pool2dOp>(p_h, p_w, p_stride, p_stride,
                                                   PoolType::AVG, output_5b);

    g.add_op("pool5/7x7_s1", pool5_7x7_s1, group_id);

    auto flatten = std::make_shared<FlattenOp>(pool5_7x7_s1);
    g.add_op("flatten", flatten, group_id);

    int num_classes = 1000;
    auto loss3_classifier = std::make_shared<AffineOp>(num_classes, flatten);
    g.add_op("loss3/classifier", loss3_classifier, group_id);

    auto softm = std::make_shared<SoftMaxOp>(loss3_classifier);
    g.add_op("prob", softm, group_id);
}
