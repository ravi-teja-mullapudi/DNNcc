#include "Graph.h"

std::shared_ptr<Op>
add_conv_relu(Graph& g, int num_filters, int filter_height,
              int filter_width, int stride, std::string suffix,
              std::shared_ptr<Op> in, int group_id)
{
    auto conv = std::make_shared<Conv2dOp>(num_filters, filter_height, filter_width,
                                           stride, stride, in);
    g.add_op("conv" + suffix, conv, group_id);

    float slope = 0.0f;
    auto relu = std::make_shared<ReLUOp>(slope, conv);
    g.add_op("relu" + suffix, relu, group_id);

    return relu;
}

void Vgg16(Graph& g) {

    // Network structure
    // input -> conv1_1 -> relu1_1 -> conv1_2 -> relu1_2 -> pool1 ->
    // conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> pool2 ->
    // conv3_1 -> relu3_1 -> conv3_2 -> relu3_2 -> conv3_3 -> relu3_3 -> pool3 ->
    // conv4_1 -> relu4_1 -> conv4_2 -> relu4_2 -> conv4_3 -> relu4_3 -> pool4 ->
    // conv5_1 -> relu5_1 -> conv5_2 -> relu5_2 -> conv5_3 -> relu5_3 -> pool5 ->
    // fc6-> relu6 -> drop6-> fc7 -> relu7 -> drop7 -> fc8 -> prob

    int group_id = g.add_group();

    int batch_size(64), channels(3), data_height(224), data_width(224);
    auto data_sizes = {batch_size, channels, data_height, data_width};
    auto data = std::make_shared<DataOp>(data_sizes);
    g.add_op("input", data, group_id);

    int num_filters(64), filter_height(3), filter_width(3), stride(1);

    auto relu1_1 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "1_1", data, group_id);

    auto relu1_2 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "1_2", relu1_1, group_id);

    int pool_height(2), pool_width(2), pool_stride(2);

    auto pool1 = std::make_shared<Pool2dOp>(pool_height, pool_width, pool_stride,
                                            pool_stride, PoolType::MAX, relu1_2);

    g.add_op("pool1", pool1, group_id);

    num_filters = 128;

    auto relu2_1 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "2_1", pool1, group_id);

    auto relu2_2 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "2_2", relu2_1, group_id);

    auto pool2 = std::make_shared<Pool2dOp>(pool_height, pool_width, pool_stride,
                                            pool_stride, PoolType::MAX, relu2_2);
    g.add_op("pool2", pool2, group_id);

    num_filters = 256;

    auto relu3_1 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "3_1", pool2, group_id);

    auto relu3_2 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "3_2", relu3_1, group_id);

    auto relu3_3 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "3_3", relu3_2, group_id);

    auto pool3 = std::make_shared<Pool2dOp>(pool_height, pool_width, pool_stride,
                                            pool_stride, PoolType::MAX, relu3_3);
    g.add_op("pool3", pool3, group_id);

    num_filters = 512;

    auto relu4_1 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "4_1", pool3, group_id);

    auto relu4_2 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "4_2", relu4_1, group_id);

    auto relu4_3 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "4_3", relu4_2, group_id);

    auto pool4 = std::make_shared<Pool2dOp>(pool_height, pool_width, pool_stride,
                                            pool_stride, PoolType::MAX, relu4_3);
    g.add_op("pool4", pool4, group_id);

    num_filters = 512;

    auto relu5_1 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "5_1", pool4, group_id);

    auto relu5_2 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "5_2", relu5_1, group_id);

    auto relu5_3 = add_conv_relu(g, num_filters, filter_height, filter_width,
                                 stride, "5_3", relu5_2, group_id);

    auto pool5 = std::make_shared<Pool2dOp>(pool_height, pool_width, pool_stride,
                                            pool_stride, PoolType::MAX, relu5_3);
    g.add_op("pool5", pool5, group_id);

    auto flatten = std::make_shared<FlattenOp>(pool5);
    g.add_op("flatten", flatten, group_id);

    int fc6_out_dim = 4096;
    auto fc6 = std::make_shared<AffineOp>(fc6_out_dim, flatten);
    g.add_op("fc6", fc6, group_id);

    float slope = 0.0f;
    auto relu6 = std::make_shared<ReLUOp>(slope, fc6);
    g.add_op("relu6", relu6, group_id);

    // TODO: add drop out for completeness. dropout is a passthrough
    // in the forward pass.

    int fc7_out_dim = 4096;
    auto fc7 = std::make_shared<AffineOp>(fc7_out_dim, relu6);
    g.add_op("fc7", fc7, group_id);

    auto relu7 = std::make_shared<ReLUOp>(slope, fc7);
    g.add_op("relu7", relu7, group_id);

    // TODO: add drop out for completeness. dropout is a passthrough
    // in the forward pass.

    int num_classes = 1000;
    auto fc8 = std::make_shared<AffineOp>(num_classes, relu7);
    g.add_op("fc8", fc8, group_id);

    auto prob = std::make_shared<SoftMaxOp>(fc8);
    g.add_op("prob", prob, group_id);
}
