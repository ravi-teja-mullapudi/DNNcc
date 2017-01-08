from __future__ import print_function
import cgen as c

data_types = []
def gen_macro(op_name, attrs, inputs, outputs):
    return "blah"

def generate_op_registration(op_name, attrs, inputs, outputs):
    # Add the headers into the module
    contents = []
    contents.append(c.Include("tensorflow/core/framework/op.h", system = False))
    contents.append(c.Include("tensorflow/core/framework/shape_inference.h", system = False))

    # Name space declarations
    contents.append(c.Line())
    contents.append(c.Statement("using namespace tensorflow"))
    contents.append(c.Line())

    # Registration macro
    reg_macro = gen_macro(op_name, attrs, inputs, outputs)
    contents.append(c.Statement(reg_macro))
    return c.Module(contents)

def generate_op_kernel():
    return

def generate_op_python_wrapper():
    return

def generate_op_gradient():
    return

print(generate_op_registration("blah", {}, {}, {}))
