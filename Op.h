#include <vector>
#include <cmath>
#include <memory>
#include "NDArray.h"

/* A node in the dataflow graph which performs an operation */
class Op {
    public:
    // Ordered list of op which create inputs for the current op.
    std::vector<std::shared_ptr<Op>> inputs;

    // Ordered list of learnable parameters of the op.
    std::vector<NDArray<float>> params;

    // Ordered list of parameter gradients of the op.
    std::vector<NDArray<float>> param_grads;

    Op() {}

    Op(const std::vector<std::shared_ptr<Op>>& _inputs) {
        for (size_t i = 0; i < _inputs.size(); i++) {
            inputs.push_back(_inputs[i]);
        }
    }

    virtual ~Op() {}

    virtual int num_outputs() = 0;
    virtual int num_dims(int out_id) = 0;
    virtual int out_size(int out_id, int dim_id) = 0;
};

class AffineOp: public Op {
    public:
    int batch_size;
    int num_inputs;
    int num_units;

    AffineOp(int _num_units, std::shared_ptr<Op> _input)
             : Op({_input}), num_units(_num_units)
    {
        assert(_input->num_outputs() == 1);
        assert(_input->num_dims(0) == 2);
        batch_size = _input->out_size(0, 0);
        num_inputs = _input->out_size(0, 1);

        params.push_back(NDArray<float>({num_units, num_inputs}));
        params.push_back(NDArray<float>({num_units}));

        // TODO: Create buffers for gradients
    }

    int num_outputs() { return 1; }

    int num_dims(int out_id) {
        assert(out_id < 1);
        return 2;
    }

    int out_size(int out_id, int dim_id) {
        assert(out_id < 1 && dim_id < 2);
        int size = 0;
        if (dim_id == 0) {
            size = batch_size;
        } else {
            size = num_inputs;
        }
        return size;
    }
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

    Conv2dOp(int _output_channels,
             int _filter_height,
             int _filter_width,
             int _stride_h,
             int _stride_w,
             std::shared_ptr<Op> _input)
        : Op({_input}),
          output_channels(_output_channels),
          filter_height(_filter_height),
          filter_width(_filter_width),
          stride_h(_stride_h),
          stride_w(_stride_w)
    {
        assert(_input->num_outputs() == 1);
        assert(_input->num_dims(0) == 4);

        batch_size = _input->out_size(0, 0);
        input_channels = _input->out_size(0, 1);
        input_height = _input->out_size(0, 2);
        input_width = _input->out_size(0, 3);

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

    int num_outputs() { return 1; }

    int num_dims(int out_id) {
        assert(out_id < 1);
        return 4;
    }

    int out_size(int out_id, int dim_id) {
        assert(out_id < 1 && dim_id < 4);
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

    Pool2dOp(int _pool_height,
             int _pool_width,
             int _stride_h,
             int _stride_w,
             PoolType _pool_type,
             std::shared_ptr<Op> _input)
        : Op({_input}),
          pool_height(_pool_height),
          pool_width(_pool_width),
          stride_h(_stride_h),
          stride_w(_stride_w),
          pool_type(_pool_type)
    {
        assert(_input->num_outputs() == 1);
        assert(_input->num_dims(0) == 4);

        batch_size = _input->out_size(0, 0);
        input_channels = _input->out_size(0, 1);
        input_height = _input->out_size(0, 2);
        input_width = _input->out_size(0, 3);

        pad_h = (pool_height - 1)/2;
        pad_w = (pool_width - 1)/2;

        output_height = 1 +
            std::ceil((float)(input_height + 2 * pad_h - pool_height)/stride_h);
        output_width = 1 +
            std::ceil((float)(input_width + 2 * pad_w - pool_width)/stride_w);
    }

    int num_outputs() { return 1; }

    int num_dims(int out_id) {
        assert(out_id < 1);
        return 4;
    }

    int out_size(int out_id, int dim_id) {
        assert(out_id < 1 && dim_id < 4);
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
};

class ReLUOp: public Op {
    public:
    float slope;
    ReLUOp(float _slope, std::shared_ptr<Op> _input)
        : Op({_input}),
          slope(_slope) {}

    int num_outputs() { return 1; }

    int num_dims(int out_id) {
        assert(out_id < 1);
        return inputs[0]->num_dims(out_id);
    }

    int out_size(int out_id, int dim_id) {
        assert(out_id < 1 && dim_id < inputs[0]->num_dims(out_id));
        return inputs[0]->out_size(out_id, dim_id);
    }
};

class SoftMaxOp: public Op {
    public:
    int batch_size;
    int num_classes;

    SoftMaxOp(std::shared_ptr<Op> _input)
        : Op({_input}) {
        assert(_input->num_outputs() == 1);
        assert(_input->num_dims(0) == 2);
        batch_size = _input->out_size(0, 0);
        num_classes = _input->out_size(0, 1);
    }

    int num_outputs() { return 1; }

    int num_dims(int out_id) {
        assert(out_id < 1);
        return 2;
    }

    int out_size(int out_id, int dim_id) {
        assert(out_id < 1 && dim_id < 2);
        int size = 0;
        if (dim_id == 0) {
            size = batch_size;
        } else {
            size = num_classes;
        }
        return size;
    }
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

    LRNOp(int _window_size, float _alpha, float _beta,
          std::shared_ptr<Op> _input)
        : Op({_input}),
          window_size(_window_size),
          alpha(_alpha),
          beta(_beta)
    {
        assert(_input->num_outputs() == 1);
        assert(_input->num_dims(0) == 2);
        batch_size = _input->out_size(0, 0);
        input_channels = _input->out_size(0, 1);
        input_height = _input->out_size(0, 2);
        input_width = _input->out_size(0, 3);
    }

    int num_outputs() { return 1; }

    int num_dims(int out_id) {
        assert(out_id < 1);
        return 4;
    }

    int out_size(int out_id, int dim_id) {
        assert(out_id < 1 && dim_id < 4);
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
};

class ConcatOp: public Op {
    public:
    int batch_size;
    int output_channels;
    int input_height;
    int input_width;

    ConcatOp(std::vector<std::shared_ptr<Op>>& _inputs)
        : Op(_inputs)
    {
        assert(inputs.size() > 0);
        assert(inputs[0]->num_outputs() == 1 && inputs[0]->num_dims(0) == 4);
        batch_size = inputs[0]->out_size(0, 0);
        input_height = inputs[0]->out_size(0, 2);
        input_width = inputs[0]->out_size(0, 3);

        output_channels = 0;
        for (size_t l = 0; l < inputs.size(); l++) {
            assert(inputs[l]->num_outputs() == 1);
            assert(inputs[l]->out_size(0, 0) == batch_size &&
                   inputs[l]->out_size(0, 2) == input_height &&
                   inputs[l]->out_size(0, 3) == input_width);
            output_channels += inputs[l]->out_size(0, 1);
        }
    }

    int num_outputs() { return 1; }

    int num_dims(int out_id) {
        assert(out_id < 1);
        return 4;
    }

    int out_size(int out_id, int dim_id) {
        assert(out_id < 1 && dim_id < 4);
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
};

class FlattenOp: public Op {
    public:
    int batch_size;
    int output_width;

    FlattenOp(std::shared_ptr<Op> _input) : Op({_input}) {
        assert(inputs[0]->num_outputs() == 1);
        assert(inputs[0]->num_dims(0) >= 2 && inputs[0]->num_dims(0) <= 4);
        batch_size = inputs[0]->out_size(0, 0);
        if (inputs[0]->num_dims(0) == 2) {
            output_width = inputs[0]->out_size(0, 1);
        } else if (inputs[0]->num_dims(0) == 3) {
            output_width = inputs[0]->out_size(0, 1) *
                           inputs[0]->out_size(0, 2);
        } else if (inputs[0]->num_dims(0) == 4) {
            output_width = inputs[0]->out_size(0, 1) *
                           inputs[0]->out_size(0, 2) *
                           inputs[0]->out_size(0, 3);
        }
    }

    int num_outputs() { return 1; }

    int num_dims(int out_id) {
        assert(out_id < 1);
        return 2;
    }

    int out_size(int out_id, int dim_id) {
        assert(out_id < 1 && dim_id < 4);
        int size = 0;
        if (dim_id == 0) {
            size = batch_size;
        } else if (dim_id == 1) {
            size = output_width;
        }
        return size;
    }
};

class DataOp: public Op {
    public:
    int batch_size;
    int input_channels;
    int input_height;
    int input_width;

    DataOp(int _batch_size,
           int _input_channels,
           int _input_height,
           int _input_width) :
        Op(),
        batch_size(_batch_size),
        input_channels(_input_channels),
        input_height(_input_height),
        input_width(_input_width) {}

    int num_outputs() { return 1; }

    int num_dims(int out_id) {
        assert(out_id < 1);
        return 4;
    }

    int out_size(int out_id, int dim_id) {
        assert(out_id < 1 && dim_id < 4);
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
};
