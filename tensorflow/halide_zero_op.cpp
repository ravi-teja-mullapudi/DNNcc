#include "Halide.h"

using namespace Halide;

int main() {
    ImageParam input{Int(32), 1};

    Var x;
    Func f("zero_op");
    f(x) = select(x == 0, input(0), 0);

    Target t = get_target_from_environment();

    f.compile_to_static_library("gen_halide_zero_op",
                                {input},
                                "halide_zero_op",
                                t);
    return 0;
}
