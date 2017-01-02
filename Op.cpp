#include "Op.h"
#include<iostream>

NDArray_t get_ndarray_t(const std::vector<int>& sizes, DataType type) {

    switch(type) {
        case DataType::Float64:
            return NDArray<double>(sizes);
        case DataType::Float32:
            return NDArray<float>(sizes);
        case DataType::Int64:
            return NDArray<int64_t>(sizes);
        case DataType::Int32:
            return NDArray<int32_t>(sizes);
        case DataType::Int16:
            return NDArray<int16_t>(sizes);
        case DataType::Int8:
            return NDArray<int8_t>(sizes);
        case DataType::UInt64:
            return NDArray<uint64_t>(sizes);
        case DataType::UInt32:
            return NDArray<uint32_t>(sizes);
        case DataType::UInt16:
            return NDArray<uint16_t>(sizes);
        case DataType::UInt8:
            return NDArray<uint8_t>(sizes);
        default:
            assert(0);
    }
}

AffineOp::AffineOp(int _num_units, std::shared_ptr<Op> _input_op)
                   : Op({_input_op}), num_units(_num_units)
{
    assert(_input_op->num_dims() == 2);
    batch_size = _input_op->out_size(0);
    num_inputs = _input_op->out_size(1);
    type = DataType::Float32;

    params.push_back(get_ndarray_t({num_units, num_inputs}, type));
    params.push_back(get_ndarray_t({num_units}, type));
    // TODO: Create buffers for gradients
}

Conv2dOp::Conv2dOp(int _output_channels,
                   int _filter_height,
                   int _filter_width,
                   int _stride_h,
                   int _stride_w,
                   std::shared_ptr<Op> _input_op,
                   bool _bias)
                   : Op({_input_op}),
                    output_channels(_output_channels),
                    filter_height(_filter_height),
                    filter_width(_filter_width),
                    stride_h(_stride_h),
                    stride_w(_stride_w),
                    bias(_bias)
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

    type = DataType::Float32;

    params.push_back(get_ndarray_t({output_channels,
                                  input_channels,
                                  filter_height,
                                  filter_width}, type));
    if (bias) {
        params.push_back(get_ndarray_t({output_channels}, type));
    }
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

    type = DataType::Float32;
}

ReLUOp::ReLUOp(float _slope, std::shared_ptr<Op> _input_op)
               : Op({_input_op}),
                 slope(_slope) {

    type = DataType::Float32;
}

SoftMaxOp::SoftMaxOp(std::shared_ptr<Op> _input_op)
                    : Op({_input_op})
{
    assert(_input_op->num_dims() == 2);
    batch_size = _input_op->out_size(0);
    num_classes = _input_op->out_size(1);
    type = DataType::Float32;
}

LRNOp::LRNOp(int _window_size, float _alpha, float _beta,
             std::shared_ptr<Op> _input_op)
             : Op({_input_op}),
               window_size(_window_size),
               alpha(_alpha),
               beta(_beta)
{
    assert(_input_op->num_dims() == 4);
    batch_size = _input_op->out_size(0);
    input_channels = _input_op->out_size(1);
    input_height = _input_op->out_size(2);
    input_width = _input_op->out_size(3);
    type = DataType::Float32;
}

BNCaffeOp::BNCaffeOp(float _epsilon, std::shared_ptr<Op> _input_op)
          : Op({_input_op}),
            epsilon(_epsilon)
{
    assert(_input_op->num_dims() == 4);
    batch_size = _input_op->out_size(0);
    output_channels = _input_op->out_size(1);
    output_height = _input_op->out_size(2);
    output_width = _input_op->out_size(3);
    type = DataType::Float32;
}

ScaleCaffeOp::ScaleCaffeOp(std::shared_ptr<Op> _input_op)
             : Op({_input_op})
{
    assert(_input_op->num_dims() == 4);
    batch_size = _input_op->out_size(0);
    output_channels = _input_op->out_size(1);
    output_height = _input_op->out_size(2);
    output_width = _input_op->out_size(3);
    type = DataType::Float32;
}

ConcatOp::ConcatOp(std::vector<std::shared_ptr<Op>>& _input_ops)
                   : Op(_input_ops)
{
    assert(input_ops.size() > 0);
    assert(input_ops[0]->num_dims() == 4);
    batch_size = input_ops[0]->out_size(0);
    input_height = input_ops[0]->out_size(2);
    input_width = input_ops[0]->out_size(3);
    type = DataType::Float32;

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
    type = DataType::Float32;
}

DataOp::DataOp(const std::vector<int>& _dim_sizes)
               : Op(),
                 dim_sizes(_dim_sizes) {

    type = DataType::Float32;
}

SumOp::SumOp(std::vector<std::shared_ptr<Op>>& _input_ops)
             : Op(_input_ops) {

    type = DataType::Float32;

    for (int d = 0; d < input_ops[0]->num_dims(); d++) {
        dim_sizes.push_back(input_ops[0]->out_size(d));
    }

    for (size_t i = 1; i < input_ops.size(); i++) {
        assert(input_ops[i]->num_dims() == (int)dim_sizes.size());
        for (int d = 0; d < input_ops[i]->num_dims(); d++) {
            assert(input_ops[i]->out_size(d) == dim_sizes[d]);
        }
    }
}
