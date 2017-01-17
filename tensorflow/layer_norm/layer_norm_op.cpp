#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "gen_halide_ln_2d.h"
#include "gen_halide_ln_4d.h"

using namespace tensorflow;

#include "tf_halide_utils.h"

REGISTER_OP("Lnorm")
.Input("input: float")
.Input("gamma: float")
.Input("beta: float")
.Output("norm: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
});

class LnormOp: public OpKernel {
    public:
    explicit LnormOp(OpKernelConstruction* context) : OpKernel(context) {

    }

    void Compute(OpKernelContext* context) override
    {
        // Get tf buffers and create halide buffers and
        // call the function with them
        const Tensor& in = context->input(0);
        const Tensor& beta = context->input(1);
        const Tensor& gamma = context->input(2);

        Tensor* norm = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, in.shape(), &norm));

        buffer_t in_buf = {0};
        init_halide_buf(in_buf, in);

        buffer_t gamma_buf = {0};
        init_halide_buf(gamma_buf, gamma);

        buffer_t beta_buf = {0};
        init_halide_buf(beta_buf, beta);

        buffer_t norm_buf = {0};
        init_halide_buf(norm_buf, norm);

        if (in.shape().dims() == 2) {
            halide_ln_2d(&in_buf, &beta_buf, &gamma_buf, 1e-12, &norm_buf);
        } else if (in.shape().dims() == 4) {
            halide_ln_4d(&in_buf, &beta_buf, &gamma_buf, 1e-12, &norm_buf);
        } else {
            assert(0);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("Lnorm").Device(DEVICE_CPU), LnormOp);
