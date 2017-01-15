#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "gen_halide_bn_inference.h"

using namespace tensorflow;

REGISTER_OP("Bnorm")
.Input("input: float")
.Input("gamma: float")
.Input("beta: float")
.Input("mean: float")
.Input("var: float")
.Output("norm: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
});

void init_halide_buf(buffer_t& buf, Tensor* t) {

    assert(t->shape().dims() <= 4);

    int curr_stride = 1;
    for (size_t d = 0; d < t->shape().dims(); d++) {
        buf.stride[d] = curr_stride;
        buf.min[d] = 0;
        buf.extent[d] = t->shape().dim_size(d);
    }

    assert(t->dtype() == DT_FLOAT);

    switch(t->shape().dims()) {
        case 1:
            buf.host = (uint8_t*)t->tensor<float, 1>().data();
            break;
        case 2:
            buf.host = (uint8_t*)t->tensor<float, 2>().data();
            break;
        case 3:
            buf.host = (uint8_t*)t->tensor<float, 3>().data();
            break;
        case 4:
            buf.host = (uint8_t*)t->tensor<float, 4>().data();
            break;
        default:
            assert(0);
    }

    buf.elem_size = 4;
}

void init_halide_buf(buffer_t& buf, const Tensor& t) {
    assert(t.shape().dims() <= 4);
    int curr_stride = 1;
    for (size_t d = 0; d < t.shape().dims(); d++) {
        buf.stride[d] = curr_stride;
        buf.min[d] = 0;
        buf.extent[d] = t.shape().dim_size(d);
    }

    assert(t.dtype() == DT_FLOAT);

    switch(t.shape().dims()) {
        case 1:
            buf.host = (uint8_t*)t.tensor<float, 1>().data();
            break;
        case 2:
            buf.host = (uint8_t*)t.tensor<float, 2>().data();
            break;
        case 3:
            buf.host = (uint8_t*)t.tensor<float, 3>().data();
            break;
        case 4:
            buf.host = (uint8_t*)t.tensor<float, 4>().data();
            break;
        default:
            assert(0);
    }

    buf.elem_size = 4;
}

class BnormOp: public OpKernel {
    public:
    explicit BnormOp(OpKernelConstruction* context) : OpKernel(context) {

    }

    void Compute(OpKernelContext* context) override
    {
        // Get tf buffers and create halide buffers and
        // call the function with them
        const Tensor& in = context->input(0);
        const Tensor& gamma = context->input(1);
        const Tensor& beta = context->input(2);
        const Tensor& mean = context->input(3);
        const Tensor& var = context->input(4);

        Tensor* norm = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, in.shape(), &norm));

        buffer_t in_buf = {0};
        init_halide_buf(in_buf, in);

        buffer_t gamma_buf = {0};
        init_halide_buf(gamma_buf, gamma);

        buffer_t beta_buf = {0};
        init_halide_buf(beta_buf, beta);

        buffer_t mean_buf = {0};
        init_halide_buf(mean_buf, mean);

        buffer_t var_buf = {0};
        init_halide_buf(var_buf, var);

        buffer_t norm_buf = {0};
        init_halide_buf(norm_buf, norm);

        halide_bn_inference(&in_buf, &mean_buf, &var_buf,
                            &beta_buf, &gamma_buf, 1e-04,
                            &norm_buf);
    }
};

REGISTER_KERNEL_BUILDER(Name("Bnorm").Device(DEVICE_CPU), BnormOp);
