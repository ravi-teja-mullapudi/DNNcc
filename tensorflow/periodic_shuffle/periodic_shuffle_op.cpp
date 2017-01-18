#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "gen_periodic_shuffle.h"

using namespace tensorflow;
using namespace tensorflow::shape_inference;

#include "tf_halide_utils.h"

REGISTER_OP("PeriodicShuffle")
.Input("input: float")
.Attr("r: int")
.Output("shuffle: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

        ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

        int r;
        TF_RETURN_IF_ERROR(c->GetAttr("r", &r));

        DimensionHandle batch_size = c->Dim(input_shape, 0);

        DimensionHandle out_height;
        TF_RETURN_IF_ERROR(c->Multiply(c->Dim(input_shape, 1),
                                       r, &out_height));

        DimensionHandle out_width;
        TF_RETURN_IF_ERROR(c->Multiply(c->Dim(input_shape, 2),
                                       r, &out_width));

        DimensionHandle out_channels;
        TF_RETURN_IF_ERROR(c->Divide(c->Dim(input_shape, 3),
                                     r * r, true, &out_channels));

        ShapeHandle output_shape;

        output_shape = c->MakeShape({batch_size,
                                     out_height,
                                     out_width,
                                     out_channels});

        c->set_output(0, output_shape);
        return Status::OK();
});

class PeriodicShuffleOp: public OpKernel {

    public:
    explicit PeriodicShuffleOp(OpKernelConstruction* context)
             : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("r", &r_));
    }

    void Compute(OpKernelContext* context) override
    {
        // Get tf buffers and create halide buffers and
        // call the function with them
        const Tensor& in = context->input(0);

        assert(in.shape().dims() == 4);
        assert(in.shape().dim_size(3) % (r_ * r_)  == 0);

        TensorShape out_shape{in.shape().dim_size(0),
                              in.shape().dim_size(1) * r_,
                              in.shape().dim_size(2) * r_,
                              in.shape().dim_size(3) / (r_ * r_)};
        Tensor* shuffle = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &shuffle));

        buffer_t in_buf = {0};
        init_halide_buf(in_buf, in);

        buffer_t shuffle_buf = {0};
        init_halide_buf(shuffle_buf, shuffle);

        halide_periodic_shuffle(&in_buf, r_, &shuffle_buf);
    }

    private:
    int r_;
};

REGISTER_KERNEL_BUILDER(Name("PeriodicShuffle").Device(DEVICE_CPU), PeriodicShuffleOp);
