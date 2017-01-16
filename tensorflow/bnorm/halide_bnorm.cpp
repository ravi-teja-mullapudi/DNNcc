#include "Halide.h"

using namespace Halide;

int main() {
    ImageParam input(Float(32), 4);
    ImageParam mean(Float(32), 1);
    ImageParam var(Float(32), 1);
    ImageParam beta(Float(32), 1);
    ImageParam gamma(Float(32), 1);

    Param<float> eps;

    Var x, y, z, n;
    Func norm("norm");
    Expr std_inv = sqrt(1.0f/(var(z) + eps));
    norm(x, y, z, n) = ((input(x, y, z, n) - mean(z)) *
                        (std_inv * gamma(z))) + beta(z);

    // Schedule
    norm.vectorize(x, 8).parallel(n);

    Target t = get_target_from_environment();

    norm.compile_to_static_library("gen_halide_bn",
                                    {input, mean, var, beta, gamma, eps},
                                    "halide_bn",
                                    t);
    return 0;
}
