#include "OpHalide.h"

void affine_forward_halide(std::string name,
                           std::shared_ptr<AffineOp> op,
                           Func input,
                           std::shared_ptr<HalideOpImpl> op_impl,
                           TargetArch arch)
{
    check_defined(input);
    Func forward(name + "_forward");

    ImageParam W(Float(32), 2);
    ImageParam b(Float(32), 1);

    op_impl->params.push_back(W);
    op_impl->params.push_back(b);

    RDom r(0, op->num_inputs);

    Var unit_dim, n;
    forward(unit_dim, n) = b(unit_dim);
    forward(unit_dim, n) = input(r.x, n) * W(r.x, unit_dim);

    if (arch == TargetArch::CPU) {
        forward.compute_root();
        if (op->batch_size > 1) {
            forward.parallel(n);
            forward.update().parallel(n);
        }
    } else if (arch == TargetArch::GPU) {
        assert(0);
    }

    forward.bound(unit_dim, 0, op->num_inputs)
           .bound(n, 0, op->batch_size);

    op_impl->output = forward;
}

void conv2d_forward_halide(std::string name,
                           std::shared_ptr<Conv2dOp> op,
                           Func input,
                           std::shared_ptr<HalideOpImpl> op_impl,
                           TargetArch arch)
{
    check_defined(input);
    Func in_bound(name + "_bound");
    in_bound = BoundaryConditions::constant_exterior(input, 0,
                                                     0, op->input_width,
                                                     0, op->input_height);

    ImageParam W(Float(32), 4);
    ImageParam b(Float(32), 1);

    op_impl->params.push_back(W);
    op_impl->params.push_back(b);

    Func forward(name + "_forward");
    Func stage(name + "_stage");
    Func W_f(name + "_W_f");

    int stride_w = op->stride_w;
    int stride_h = op->stride_h;

    int pad_w = op->pad_w;
    int pad_h = op->pad_h;

    Var x, y, z, n;
    W_f(x, y, z, n) = W(x, y, z, n);
    stage(x, y, z, n) = b(z);
    stage(x, y, z, n) = W_f(r.x, r.y, r.z, z) *
                        in_bound(x * stride_w + r.x - pad_w,
                                 y * stride_h + r.y - pad_h,
                                 r.z, n);

    forward(x, y, z, n) = stage(x, y, z, n);

    if (arch == TargetArch::CPU) {
        in_bound.compute_root();
        forward.compute_root();
        if (op->batch_size > 1) {
            forward.parallel(n);
        }

        if (op->output_channels > 1) {
            forward.parallel(z);
        }

        forward.vectorize(x, 8);

    } else (arch == TargetArch::GPU) {
        assert(0);
    }

    forward.bound(x, 0, op->output_width)
           .bound(y, 0, op->output_height)
           .bound(z, 0, op->output_channels)
           .bound(n, 0, op->batch_size);

    op_impl->output = forward;
}

void pool2d_forward_halide(std::string name,
                           std::shared_ptr<Pool2dOp> op,
                           PoolType pool_type,
                           Func input,
                           std::shared_ptr<HalideOpImpl> op_impl,
                           TargetArch arch)
{
    check_defined(input);
    Func in_bound(name + "_bound");
    in_bound = BoundaryConditions::constant_exterior(input, 0,
                                                     0, op->input_width,
                                                     0, op->input_height);
    int p_w = op->pool_width;
    int p_h = op->pool_height;

    int stride_w = op->stride_w;
    int stride_h = op->stride_h;

    int pad_w = op->pad_w;
    int pad_h = op->pad_h;

    Func forward(name + "_forward");
    Func stage(name + "_stage");

    Var x, y, z, n;

    RDom r(0, p_w, 0, p_h);
    if (pool_type == PoolType::MAX) {
        stage(x, y, z, n) = Float(32).min();
        stage(x, y, z, n) = max(in_bound(x * stride_w + r.x - pad_w,
                                         y * stride_h + r.y - pad_h,
                                         z, n), stage(x, y, z, n));
        forward(x, y, z, n) = stage(x, y, z, n);
    } else if (pool_type == PoolType::AVG) {
        stage(x, y, z, n) = 0.0f;
        stage(x, y, z, n) = in_bound(x * stride_w + r.x - pad_w,
                                     y * stride_h + r.y - pad_h,
                                     z, n);
        forward(x, y, z, n) = stage(x, y, z, n)/(p_w * p_h);
    }

    if (arch == TargetArch::CPU) {
        if (op->batch_size > 1) {
            forward.parallel(n);
        }
        forward.vectorize(x, 8);
    } else (arch == TargetArch::GPU) {
        assert(0);
    }

    forward.bound(x, 0, op->output_width)
           .bound(y, 0, op->output_height)
           .bound(z, 0, op->output_channels)
           .bound(n, 0, op->batch_size);

    op_impl->output = forward;
}

void relu_forward_halide(std::string name,
                         std::shared_ptr<ReLUOp> op,
                         Func input,
                         std::shared_ptr<HalideOpImpl> op_impl,
                         TargetArch arch)
{
    check_defined(input);
    Var x, y, z, n;
    Func forward(name + "_forward");
    switch(op->inputs[0]->out_dims()) {
        case 1:
            if (op->slope == 0.0f) {
                forward(x) = select(input(x) > 0, input(x), 0);
            } else {
                forward(x) = select(input(x) > 0, input(x), slope * input(x));
            }
            break;
        case 2:
            if (op->slope == 0.0f) {
                forward(x, y) = select(input(x, y) > 0, input(x, y), 0);
            } else {
                forward(x, y) = select(input(x, y) > 0, input(x, y),
                                                        slope * input(x, y));
            }
            break;
        case 3:
            if (op->slope == 0.0f) {
                forward(x, y, z) = select(input(x, y, z) > 0, input(x, y, z), 0);
            } else {
                forward(x, y, z) = select(input(x, y, z) > 0,
                                          input(x, y, z),
                                          slope * input(x, y, z));
            }
            break;
        case 4:
            if (op->slope == 0.0f) {
                forward(x, y, z, w) = select(input(x, y, z, w) > 0,
                                             input(x, y, z, w), 0);
            } else {
                forward(x, y, z, w) = select(input(x, y, z, w) > 0,
                                             input(x, y, z, w),
                                             slope * input(x, y, z, w));
            }
            break;
        default:
            std::cerr << "ReLU layer does not support inputs with more\
                          than 4 dimensions" << std::endl;
    }
    op_impl->output = forward;
}

void softmax_forward_halide(std::string name,
                            std::shared_ptr<SoftMaxOp> op,
                            Func input,
                            std::shared_ptr<HalideOpImpl> op_impl,
                            TargetArch arch)
{
    check_defined(input);

    Func forward(name + "_forward");
    Func exp_max(name + "_exp_max");
    Func expo(name + "_expo");
    Func normalizer(name + "_normalizer");

    Var in_dim, n;

    RDom r(0, op->num_classes);

    exp_max(n) = maximum(input(r.x, n));
    expo(in_dim, n) = exp(input(in_dim, n) - exp_max(n));

    normalizer(n) = cast(input.output_types()[0], 0);
    normalizer(n) += expo(r.x, n);

    forward(in_dim, n) = expo(in_dim, n)/normalizer(n);

    if (arch == TargetArch::CPU) {
        exp_max.compute_root();
        normalizer.compute_root();
        forward.compute_root();
    } else if (arch == TargetArch::GPU) {
        assert(0);
    }

    op_impl->output = forward;
}

void lrn_forward_halide(std::string name,
                        std::shared_ptr<LRNOp> op,
                        Func input,
                        std::shared_ptr<HalideOpImpl> op_impl,
                        TargetArch arch)
{

}
