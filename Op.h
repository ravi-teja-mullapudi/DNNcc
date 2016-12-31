#pragma once

#include <vector>
#include <cmath>
#include <memory>
#include "NDArray.h"

/* An operation node in the dataflow graph */
class Op {
    public:
    // Ordered list of op which create inputs for the current op.
    std::vector<std::shared_ptr<Op>> input_ops;

    // Ordered list of learnable parameters of the op.
    std::vector<NDArray_t> params;

    // Ordered list of parameter gradients of the op.
    std::vector<NDArray_t> param_grads;

    // Data type of params, inputs, and output of the op.
    DataType type;

    Op() {}

    Op(const std::vector<std::shared_ptr<Op>>& _input_ops) {
        for (size_t i = 0; i < _input_ops.size(); i++) {
            input_ops.push_back(_input_ops[i]);
        }
    }

    virtual ~Op() {}

    virtual int num_dims() = 0;
    virtual int out_size(int dim_id) = 0;
};

class AffineOp: public Op {
    public:
    int batch_size;
    int num_inputs;
    int num_units;

    int num_dims() { return 2; }

    int out_size(int dim_id) {
        assert(dim_id < 2);
        int size = 0;
        if (dim_id == 0) {
            size = batch_size;
        } else {
            size = num_inputs;
        }
        return size;
    }

    AffineOp(int _num_units, std::shared_ptr<Op> _input_op);
};

class Conv2dOp: public Op {
    public:
    int batch_size;
    int input_channels;
    int input_height;
    int input_width;
    int output_channels;
    int output_height;
    int output_width;
    int filter_height;
    int filter_width;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;

    int num_dims() { return 4; }

    int out_size(int dim_id) {
        assert(dim_id < 4);
        int size = 0;
        if (dim_id == 0) {
            size = batch_size;
        } else if (dim_id == 1) {
            size = output_channels;
        } else if (dim_id == 2) {
            size = output_height;
        } else if (dim_id == 3) {
            size = output_width;
        }
        return size;
    }

    Conv2dOp(int _output_channels,
             int _filter_height,
             int _filter_width,
             int _stride_h,
             int _stride_w,
             std::shared_ptr<Op> _input_op);
};

enum PoolType {AVG, MAX};

class Pool2dOp: public Op {
    public:
    int batch_size;
    int input_channels;
    int input_height;
    int input_width;
    int pool_height;
    int pool_width;
    int output_height;
    int output_width;
    int stride_h;
    int stride_w;
    PoolType pool_type;
    int pad_h;
    int pad_w;

    int num_dims() { return 4; }

    int out_size(int dim_id) {
        assert(dim_id < 4);
        int size = 0;
        if (dim_id == 0) {
            size = batch_size;
        } else if (dim_id == 1) {
            size = input_channels;
        } else if (dim_id == 2) {
            size = output_height;
        } else if (dim_id == 3) {
            size = output_width;
        }
        return size;
    }

    Pool2dOp(int _pool_height,
             int _pool_width,
             int _stride_h,
             int _stride_w,
             PoolType _pool_type,
             std::shared_ptr<Op> _input_op);
};

class ReLUOp: public Op {
    public:
    float slope;

    int num_dims() {
        return input_ops[0]->num_dims();
    }

    int out_size(int dim_id) {
        assert(dim_id < input_ops[0]->num_dims());
        return input_ops[0]->out_size(dim_id);
    }

    ReLUOp(float _slope, std::shared_ptr<Op> _input_op);
};

class SoftMaxOp: public Op {
    public:
    int batch_size;
    int num_classes;

    int num_dims() { return 2; }

    int out_size(int dim_id) {
        assert(dim_id < 2);
        int size = 0;
        if (dim_id == 0) {
            size = batch_size;
        } else {
            size = num_classes;
        }
        return size;
    }

    SoftMaxOp(std::shared_ptr<Op> _input_op);
};

class LRNOp: public Op {
    public:
    int batch_size;
    int input_channels;
    int input_height;
    int input_width;
    int window_size;
    int alpha;
    int beta;

    int num_dims() { return 4; }

    int out_size(int dim_id) {
        assert(dim_id < 4);
        int size = 0;
        if (dim_id == 0) {
            size = batch_size;
        } else if (dim_id == 1) {
            size = input_channels;
        } else if (dim_id == 2) {
            size = input_height;
        } else if (dim_id == 3) {
            size = input_width;
        }
        return size;
    }

    LRNOp(int _window_size, float _alpha, float _beta,
          std::shared_ptr<Op> _input_op);
};

class ConcatOp: public Op {
    public:
    int batch_size;
    int output_channels;
    int input_height;
    int input_width;

    int num_dims() { return 4; }

    int out_size(int dim_id) {
        assert(dim_id < 4);
        int size = 0;
        if (dim_id == 0) {
            size = batch_size;
        } else if (dim_id == 1) {
            size = output_channels;
        } else if (dim_id == 2) {
            size = input_height;
        } else if (dim_id == 3) {
            size = input_width;
        }
        return size;
    }

    ConcatOp(std::vector<std::shared_ptr<Op>>& _input_ops);
};

class FlattenOp: public Op {
    public:
    int batch_size;
    int output_width;

    int num_dims() { return 2; }

    int out_size(int dim_id) {
        assert(dim_id < 4);
        int size = 0;
        if (dim_id == 0) {
            size = batch_size;
        } else if (dim_id == 1) {
            size = output_width;
        }
        return size;
    }

    FlattenOp(std::shared_ptr<Op> _input_op);
};

class DataOp: public Op {
    public:
    std::vector<int> dim_sizes;

    int num_dims() { return dim_sizes.size(); }

    int out_size(int dim_id) {
        assert(dim_id < (int)dim_sizes.size());
        return dim_sizes[dim_id];
    }

    DataOp(const std::vector<int>& _dim_sizes);
};

class SumOp: public Op {
    public:
    std::vector<int> dim_sizes;

    int num_dims() { return dim_sizes.size(); }

    int out_size(int dim_id) {
        assert(dim_id < (int)dim_sizes.size());
        return dim_sizes[dim_id];
    }

    SumOp(const std::vector<int>& _dim_sizes,
          std::vector<std::shared_ptr<Op>>& _input_ops);
};

NDArray_t get_ndarray_t(const std::vector<int>& sizes, DataType type);
