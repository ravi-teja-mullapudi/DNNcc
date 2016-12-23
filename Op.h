#include <vector>
/* A node in the dataflow graph which performs an operation */
class Op {
    public:
    Op(std::string _name, Op& _input) : name(_name) {
        inputs.push_back(_input);
    }

    Op(std::string _name, std::vector<Op>& _inputs) : name(_name) {
        for (size_t i = 0; i < _inputs.size(); i++) {
            inputs.push_back(_inputs[i]);
        }
    }

    virtual ~Op() {};

    // Op name. Must be unique in a dataflow graph. Used to lookup
    // and initialize parameters.
    std::string name;

    // Ordered list of op which create inputs for the current op.
    std::vector<Op> inputs;

    // Ordered list of learnable parameters of the op.
    std::vector<NDArray<float>> params;

    // Ordered list of parameter gradients of the op.
    std::vector<NDArray<float>> param_grads;

    // Ordered list of outputs of the op.
    std::vector<NDArray<float>> outputs;

    virtual int num_outputs() = 0;
    virtual int num_dims(int out_id) = 0;
    virtual int out_dim_size(int out_id, int dim_id) = 0;
};
