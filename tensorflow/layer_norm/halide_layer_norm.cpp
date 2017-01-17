#include "Halide.h"

using namespace Halide;

void layer_norm_2d() {

    ImageParam input(Float(32), 2);
    ImageParam beta(Float(32), 1);
    ImageParam gamma(Float(32), 1);

    Param<float> eps;

    RDom r(input.dim(0).min(), input.dim(0).extent());

    Var n, x;
    Func mean("mean");
    Expr H = input.dim(0).extent();

    Expr inv_H = 1.0f/H;
    mean(n) = 0.0f;
    mean(n) += input(r.x, n) * inv_H;

    Func var2("var2");
    var2(n) = 0.0f;
    Expr diff = input(r.x, n) - mean(n);
    var2(n) += diff * diff * inv_H;

    Func output("norm");
    Expr inv_var = 1.0f/sqrt(var2(n) + eps);
    output(x, n) = ((gamma(x) * inv_var * (input(x, n) - mean(n)))
                   - beta(x));

    Target t = get_target_from_environment();

    output.compile_to_static_library("gen_halide_ln_2d",
                                    {input, beta, gamma, eps},
                                    "halide_ln_2d",
                                    t);
}

void layer_norm_4d() {

    ImageParam input(Float(32), 4);
    ImageParam beta(Float(32), 3);
    ImageParam gamma(Float(32), 3);

    Param<float> eps;

    RDom r(input.dim(0).min(), input.dim(0).extent(),
           input.dim(1).min(), input.dim(1).extent(),
           input.dim(2).min(), input.dim(2).extent());

    Var n, z, y, x;
    Func mean("mean");
    Expr H = input.dim(0).extent() *
             input.dim(1).extent() *
             input.dim(2).extent();

    Expr inv_H = 1.0f/H;
    mean(n) = 0.0f;
    mean(n) += input(r.x, r.y, r.z, n) * inv_H;

    Func var2("var2");
    var2(n) = 0.0f;
    Expr diff = input(r.x, r.y, r.z, n) - mean(n);
    var2(n) += diff * diff * inv_H;

    Func output("output");
    Expr inv_var = 1.0f/sqrt(var2(n) + eps);

    output(x, y, z, n) =
        (gamma(x, y, z) * inv_var * (input(x, y, z, n) - mean(n)))
        - beta(x, y, z);

    mean.compute_root();
    var2.compute_root();
    output.compute_root();

    Target t = get_target_from_environment();

    output.compile_to_static_library("gen_halide_ln_4d",
                                     {input, beta, gamma, eps},
                                     "halide_ln_4d",
                                     t);
}

int main() {
    layer_norm_4d();
    layer_norm_2d();
    return 0;
}
