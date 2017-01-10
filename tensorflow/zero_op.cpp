#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "gen_halide_zero_op.h"

using namespace tensorflow;

REGISTER_OP("Zero")
.Input("to_zero: int32")
.Output("zeroed: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
{
  auto s0 = c->MakeShape({1000});
  c->set_output(0, s0);
  return Status::OK();
});

class ZeroOp: public OpKernel
{
  public:
  explicit ZeroOp(OpKernelConstruction* context) : OpKernel(context)
  {
  }

  void Compute(OpKernelContext* context) override
  {
      // Get tf buffers and create halide buffers and
      // call the function with them
      const Tensor& i0_t = context->input(0);
      Tensor* o0_t = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, {1000},
                                                       &o0_t));

      buffer_t input_buf = {0};
      input_buf.stride[0] = 1;
      input_buf.min[0] = 0;
      input_buf.extent[0] = i0_t.shape().dim_size(0);
      auto i0_data = i0_t.tensor<int32, 1>();
      input_buf.host = (uint8_t*)i0_data.data();
      input_buf.elem_size = 4;

      buffer_t output_buf = {0};
      output_buf.stride[0] = 1;
      output_buf.min[0] = 0;
      output_buf.extent[0] = o0_t->shape().dim_size(0);
      auto o0_data = o0_t->tensor<int32, 1>();
      output_buf.host = (uint8_t*)o0_data.data();
      output_buf.elem_size = 4;

      halide_zero_op(&input_buf, &output_buf);
  }
}
;

REGISTER_KERNEL_BUILDER(Name("Zero").Device(DEVICE_CPU), ZeroOp);
