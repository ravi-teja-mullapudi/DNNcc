#include "Op.h"

AffineOp::AffineOp(int _num_units, std::shared_ptr<Op> _input_op)
                   : Op({_input_op}), num_units(_num_units)
{
    assert(_input_op->num_dims() == 2);
    batch_size = _input_op->out_size(0);
    num_inputs = _input_op->out_size(1);

    params.push_back(NDArray<float>({num_units, num_inputs}));
    params.push_back(NDArray<float>({num_units}));

    // TODO: Create buffers for gradients
}

Conv2dOp::Conv2dOp(int _output_channels,
                   int _filter_height,
                   int _filter_width,
                   int _stride_h,
                   int _stride_w,
                   std::shared_ptr<Op> _input_op)
                   : Op({_input_op}),
                    output_channels(_output_channels),
                    filter_height(_filter_height),
                    filter_width(_filter_width),
                    stride_h(_stride_h),
                    stride_w(_stride_w)
{
    assert(_input_op->num_dims() == 4);

    batch_size = _input_op->out_size(0);
    input_channels = _input_op->out_size(1);
    input_height = _input_op->out_size(2);
    input_width = _input_op->out_size(3);

    pad_h = (filter_height - 1)/2;
    pad_w = (filter_width - 1)/2;

    output_height = (1 + (input_height + 2 * pad_h - filter_height)/stride_h);
    output_width = (1 + (input_width + 2 * pad_w - filter_width)/stride_w);

    params.push_back(NDArray<float>({output_channels,
                                     input_channels,
                                     filter_height,
                                     filter_width}));
    params.push_back(NDArray<float>({output_channels}));

    // TODO: Create buffers for gradients
}

Pool2dOp::Pool2dOp(int _pool_height,
                   int _pool_width,
                   int _stride_h,
                   int _stride_w,
                   PoolType _pool_type,
                   std::shared_ptr<Op> _input_op)
                   : Op({_input_op}),
                    pool_height(_pool_height),
                    pool_width(_pool_width),
                    stride_h(_stride_h),
                    stride_w(_stride_w),
                    pool_type(_pool_type)
{
    assert(_input_op->num_dims() == 4);

    batch_size = _input_op->out_size(0);
    input_channels = _input_op->out_size(1);
    input_height = _input_op->out_size(2);
    input_width = _input_op->out_size(3);

    pad_h = (pool_height - 1)/2;
    pad_w = (pool_width - 1)/2;

    // TODO: There is a discrepancy between the output widths when compared
    // to caffe. Resolve this by looking at different frameworks and doing
    // the right thing.
    output_height = 1 +
        std::ceil((float)(input_height + 2 * pad_h - pool_height)/stride_h);
    output_width = 1 +
        std::ceil((float)(input_width + 2 * pad_w - pool_width)/stride_w);
}

ReLUOp::ReLUOp(float _slope, std::shared_ptr<Op> _input_op)
               : Op({_input_op}),
                 slope(_slope) {}

SoftMaxOp::SoftMaxOp(std::shared_ptr<Op> _input_op)
                    : Op({_input_op})
{
    assert(_input_op->num_dims() == 2);
    batch_size = _input_op->out_size(0);
    num_classes = _input_op->out_size(1);
}

LRNOp::LRNOp(int _window_size, float _alpha, float _beta,
             std::shared_ptr<Op> _input_op)
             : Op({_input_op}),
               window_size(_window_size),
               alpha(_alpha),
               beta(_beta)
{
    assert(_input_op->num_dims() == 2);
    batch_size = _input_op->out_size(0);
    input_channels = _input_op->out_size(1);
    input_height = _input_op->out_size(2);
    input_width = _input_op->out_size(3);
}

ConcatOp::ConcatOp(std::vector<std::shared_ptr<Op>>& _input_ops)
                   : Op(_input_ops)
{
    assert(input_ops.size() > 0);
    assert(input_ops[0]->num_dims() == 4);
    batch_size = input_ops[0]->out_size(0);
    input_height = input_ops[0]->out_size(2);
    input_width = input_ops[0]->out_size(3);

    output_channels = 0;
    for (size_t l = 0; l < input_ops.size(); l++) {
        assert(input_ops[l]->out_size(0) == batch_size &&
                input_ops[l]->out_size(2) == input_height &&
                input_ops[l]->out_size(3) == input_width);
        output_channels += input_ops[l]->out_size(1);
    }
}

FlattenOp::FlattenOp(std::shared_ptr<Op> _input_op)
                     : Op({_input_op})
{
    assert(input_ops[0]->num_dims() >= 2 && input_ops[0]->num_dims() <= 4);
    batch_size = input_ops[0]->out_size(0);
    if (input_ops[0]->num_dims() == 2) {
        output_width = input_ops[0]->out_size(1);
    } else if (input_ops[0]->num_dims() == 3) {
        output_width = input_ops[0]->out_size(1) *
            input_ops[0]->out_size(2);
    } else if (input_ops[0]->num_dims() == 4) {
        output_width = input_ops[0]->out_size(1) *
            input_ops[0]->out_size(2) *
            input_ops[0]->out_size(3);
    }
}

DataOp::DataOp(int _batch_size,
               int _input_channels,
               int _input_height,
               int _input_width)
               : Op(),
                 batch_size(_batch_size),
                 input_channels(_input_channels),
                 input_height(_input_height),
                 input_width(_input_width) {}
