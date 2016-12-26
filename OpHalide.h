#include <string>
#include <vector>
#include "Halide.h"
#include "OpImpl.h"
#include "Op.h"

using namespace Halide;

// Sanity check to make sure the halide function is defined.
void check_defined(Func f) {
    if (!f.defined()) {
        std::cout << f.name() << " is undefined" << std::endl;
        exit(-1);
    }
}

class HalideOpImpl {
    public:
    Func output;
    std::vector<ImageParam> params;
    std::vector<Func> param_grads;
    std::vector<Func> input_grads;
};

void affine_forward_halide(std::string name,
                           std::shared_ptr<AffineOp> op,
                           Func input,
                           std::shared_ptr<HalideOpImpl> op_impl,
                           TargetArch arch);

void conv2d_forward_halide(std::string name,
                           std::shared_ptr<Conv2dOp> op,
                           Func input,
                           std::shared_ptr<HalideOpImpl> op_impl,
                           TargetArch arch);

void pool2d_forward_halide(std::string name,
                           std::shared_ptr<Pool2dOp> op,
                           PoolType pool_type,
                           Func input,
                           std::shared_ptr<HalideOpImpl> op_impl,
                           TargetArch arch);

void relu_forward_halide(std::string name,
                         std::shared_ptr<ReLUOp> op,
                         Func input,
                         std::shared_ptr<HalideOpImpl> op_impl,
                         TargetArch arch);

void softmax_forward_halide(std::string name,
                            std::shared_ptr<SoftMaxOp> op,
                            Func input,
                            std::shared_ptr<HalideOpImpl> op_impl,
                            TargetArch arch);

void lrn_forward_halide(std::string name,
                        std::shared_ptr<LRNOp> op,
                        Func input,
                        std::shared_ptr<HalideOpImpl> op_impl,
                        TargetArch arch);
