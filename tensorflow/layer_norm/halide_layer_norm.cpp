#include "Halide.h"

using namespace Halide;

void layer_norm_2d() {

    ImageParam input(Float(32), 2);
    ImageParam center(Float(32), 1);
    ImageParam scale(Float(32), 1);

    Param<float> eps;

    RDom r(input.dim(1).min(), input.dim(1).extent());

    Var n, x;
    Func mean("mean");
    Expr H = input.dim(1).extent();

    Expr inv_H = 1.0f/H;
    mean(n) = 0.0f;
    mean(n) += input(n, r.x) * inv_H;

    Func var2("var2");
    var2(n) = 0.0f;
    Expr diff = input(n, r.x) - mean(n);
    var2(n) += diff * diff * inv_H;

    Func output("norm");
    Expr inv_var = 1.0f/sqrt(var2(n));
    output(n, x) = (scale(x) * inv_var * (input(n, x) - mean(n)))
                   - center(x);

    Target t = get_target_from_environment();

    output.compile_to_static_library("gen_halide_ln_2d",
                                    {input, center, scale, eps},
                                    "halide_ln_2d",
                                    t);
}

void layer_norm_4d() {

    ImageParam input(Float(32), 4);
    ImageParam center(Float(32), 3);
    ImageParam scale(Float(32), 3);

    Param<float> eps;

    RDom r(input.dim(1).min(), input.dim(1).extent(),
           input.dim(2).min(), input.dim(2).extent(),
           input.dim(3).min(), input.dim(3).extent());

    Var n, z, y, x;
    Func mean("mean");
    Expr H = input.dim(1).extent() *
             input.dim(2).extent() *
             input.dim(3).extent();

    Expr inv_H = 1.0f/H;
    mean(n) = 0.0f;
    mean(n) += input(n, r.x, r.y, r.z) * inv_H;

    Func var2("var2");
    var2(n) = 0.0f;
    Expr diff = input(n, r.x, r.y, r.z) - mean(n);
    var2(n) += diff * diff * inv_H;

    Func output("output");
    Expr inv_var = 1.0f/sqrt(var2(n));
    output(n, z, y, x) =
        (scale(z, y, x) * inv_var * (input(n, z, y, x) - mean(n)))
        - center(z, y, x);

    Target t = get_target_from_environment();

    output.compile_to_static_library("gen_halide_ln_4d",
                                     {input, center, scale, eps},
                                     "halide_ln_4d",
                                     t);
}

int main() {
    layer_norm_4d();
    layer_norm_2d();
    return 0;
}
