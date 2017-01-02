#include "Graph.h"
std::shared_ptr<Op>
conv_bn_scale(Graph& g, std::vector<std::string>& names,
              std::shared_ptr<Op> in, int group_id,
              int num_filters, int filter_height,
              int filter_width, int stride, bool conv_bias) {

    auto conv = std::make_shared<Conv2dOp>(num_filters, filter_height, filter_width,
                                           stride, stride, data, conv_bias);
    g.add_op(names[0], conv, group_id);

    float epsilon = 1e-5;
    auto bn = std::make_shared<BNCaffeOp>(epsilon, conv);
    g.add_op(names[1], bn, group_id);

    auto scale = std::make_shared<ScaleCaffeOp>(bn);
    g.add_op(names[2], scale, group_id);

    return scale;
}

std::shared_ptr<Op>
conv_bn_scale_relu(Graph& g, std::vector<std::string>& names,
                   std::shared_ptr<Op> in, int group_id,
                   int num_filters, int filter_height,
                   int filter_width, int stride, bool conv_bias) {

    auto scale = conv_bn_scale(g, names, in, group_id, num_filters,
                               filter_height, filter_width, stride,
                               conv_bias);

    float slope = 0.0f;
    auto relu = std::make_shared<ReLUOp>(slope, scale);
    g.add_op(names[3], relu, group_id);

    return relu;
}

std::shared_ptr<Op>
residual_3unit(Graph& g, std::string name, std::shared_ptr<Op> in,
               int group_id, std::vector<int> filter_sizes,
               int downsample_stride, bool downsample) {

    std::vector<std::string> names_2a = { "res" + name + "_branch2a",
                                        "bn" + name + "_branch2a",
                                        "scale" + name + "_branch2a",
                                        "res" + name + "_branch2a_relu"};

    int stride_2a = downsample ? downsample_stride : 1;
    auto res_branch2a = conv_bn_scale_relu(g, names_2a, in, group_id,
                                          filter_sizes[0], 1, 1, stride_2a, false);

    std::vector<std::string> names_2b = { "res" + name + "_branch2b",
                                        "bn" + name + "_branch2b",
                                        "scale" + name + "_branch2b",
                                        "res" + name + "_branch2b_relu"};

    auto res_branch2b = conv_bn_scale_relu(g, names_2b, res_branch2a, group_id,
                                          filter_sizes[1], 3, 3, 1, false);

    std::vector<std::string> names_2c = { "res" + name + "_branch2c",
                                          "bn" + name + "_branch2c",
                                          "scale" + name + "_branch2c"};

    auto res_branch2c = conv_bn_scale(g, names4, res_branch2b, group_id,
                                     filter_sizes[2], 1, 1, 1, false);

    std::vector<std::shared_ptr<Op>> sum_ins = {in, res_branch2c};
    auto sum = std::make_shared<SumOp>(sum_ins);
    g.add_op(g, "res" + name, sum, group_id);

    float slope = 0.0f;
    auto relu = std::make_shared<ReLUOp>(slope, sum);
    g.add_op(g, "res" + name + "_relu", relu, group_id);

    return relu;
}

std::shared_ptr<Op>
stem(Graph& g, int group_id, int batch_size, int channels,
     int data_height, data_width) {

    auto data_sizes = {batch_size, channels, data_height, data_width};
    auto data = std::make_shared<DataOp>(data_sizes);
    g.add_op("data", data, group_id);

    int num_filters(64), filter_height(7), filter_width(7), stride(2);

    std::vector<std::string> names = {"conv1", "bn_conv1",
                                      "scale_conv1", "conv1_relu"};
    auto conv1_relu = conv_bn_scale_relu(g, "conv1", names, group_id,
                                         num_filters, filter_height,
                                         filter_width, stride, data, true);

    int p_h(3), p_w(3), p_stride(2);
    auto pool1 = std::make_shared<Pool2dOp>(p_h, p_w, p_stride, p_stride,
                                            PoolType::MAX, conv1_relu);
    g.add_op("pool1", pool1, group_id);
    return pool1;
}

std::shared_ptr<Op>
resnet_3unit(Graph& g, int& group_id, std::vector<std::vector<int>>& res_sizes,
             std::vector<std::vector<std::string>>& res_names,
             std::vector<int>& res_strides, std::shared_ptr<Op> in) {

    std::shared_ptr<Op> res_in = in;
    std::shared_ptr<Op> res_out;
    for (size_t r = 0; r < res_sizes.size(); r++) {
        std::vector<std::string> branch_names = { "res" + res_names[0] + "_branch1",
                                                  "bn" +  res_names[0] + "_branch1",
                                                  "scale" + res_names[0] + "_branch1"};
        res_out = conv_bn_scale(g, branch_names, res_in, group_id,
                                res_sizes[r][2], 1, 1, res_strides[r], false);
        res_in = res_out;
        bool downsample = true;
        for (auto &name: res_names[r]) {
            res_out = residual_3unit(g, name, res_in, group_id, res_sizes[r],
                                     res_strides[r], downsample);
            res_in = res_out;
            group_id = g.add_group();
            downsample = false;
        }
    }
    return res_out;
}

std::shared_ptr<Op>
resnet_classify(Graph& g, int group_id, std::shared_ptr<Op> in) {

    int p_w(7), p_h(7), p_sride(1);
    auto pool5 = std::make_shared<Pool2dOp>(p_h, p_w, p_stride, p_stride,
                                            PoolType::AVG, in);
    g.add_op("pool5", pool5, group_id);

    auto flatten = std::make_shared<FlattenOp>(pool5);
    g.add_op("flatten", flatten, group_id);

    int num_classes = 1000;
    auto fc1000 = std::make_shared<AffineOp>(num_classes, flatten);
    g.add_op("fc1000", fc1000, group_id);

    auto softm = std::make_shared<SoftMaxOp>(fc1000);
    g.add_op("prob", softm, group_id);

    return sotfm;
}

void Resnet18(Graph& g, int batch_size, int channels,
              int data_height, int data_width) {
    int group_id = g.add_group();
    auto pool1 = stem(g, group_id, batch_size, channels,
                      data_height, data_width);
}

void Resnet34(Graph& g, int batch_size, int channels,
              int data_height, int data_width) {
    int group_id = g.add_group();
    auto pool1 = stem(g, group_id, batch_size, channels,
                      data_height, data_width);
}

void Resnet50(Graph& g, int batch_size, int channels,
              int data_height, int data_width) {

    int group_id = g.add_group();

    auto pool1 = stem(g, group_id, batch_size, channels,
                     data_height, data_width);

    group_id = g.add_group();
    std::vector<std::vector<int>> res_sizes = { {64, 64, 256},
                                                {128, 128, 512},
                                                {256, 256, 1024},
                                                {512, 512, 2048} };

    std::vector<int> res_strides = {1, 2, 2, 2};

    std::vector<std::vector<std::string>> res_names = {{"2a", "2b", "2c"},
                                                       {"3a", "3b", "3c", "3d"},
                                                       {"4a", "4b", "4c", "4d", "4e", "4f"},
                                                       {"5a", "5b", "5c"}};

    auto res5c = resnet_3unit(g, group_id, res_sizes, res_names,
                              res_strides, pool1);

    group_id = g.add_group();

    auto prob = resnet_classify(g, group_id, res5c);
}

void Resnet101(Graph& g, int batch_size, int channels,
               int data_height, int data_width) {
    int group_id = g.add_group();
    auto pool1 = stem(g, group_id, batch_size, channels,
                      data_height, data_width);
}

void Resnet152(Graph& g, int batch_size, int channels,
               int data_height, int data_width) {
    int group_id = g.add_group();
    auto pool1 = stem(g, group_id, batch_size, channels,
                      data_height, data_width);
}
