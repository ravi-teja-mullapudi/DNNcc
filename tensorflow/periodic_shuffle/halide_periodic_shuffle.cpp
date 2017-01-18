#include "Halide.h"

using namespace Halide;

void periodic_shuffle() {
    ImageParam input(Float(32), 4);
    Param<int> r;

    Func shuffle("periodic_shuffle");

    Var x, y, z, n;

    Expr x_idx = x/r;
    Expr y_idx = y/r;

    Expr z_idx = (input.dim(0).extent()/r) * (y%r) +
                 (input.dim(0).extent()/(r * r)) * (x%r) + z;

    shuffle(z, x, y, n) = input(z_idx, x_idx, y_idx, n);

    Target t = get_target_from_environment();

    shuffle.compile_to_static_library("gen_periodic_shuffle",
                                      {input, r},
                                      "halide_periodic_shuffle",
                                      t);
}

int main() {
    periodic_shuffle();
    return 0;
}
