#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "gen_halide_bn.h"

using namespace tensorflow;

#include "tf_halide_utils.h"

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

        halide_bn(&in_buf, &mean_buf, &var_buf,
                  &beta_buf, &gamma_buf, 1e-04,
                  &norm_buf);
    }
};

REGISTER_KERNEL_BUILDER(Name("Bnorm").Device(DEVICE_CPU), BnormOp);
