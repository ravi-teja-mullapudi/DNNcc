from __future__ import print_function
import cgen as c

# attrs can have their own syntax look at
# tensorflow/core/framework/op_def_builder.h
# for more details

# List of allowed types
# tensorflow/core/framework/types.cc
attr_data_types = ["float", "double",
                   "int8", "int16", "int32", "int64",
                   "uint8", "uint16",
                   "bool",
                   "qint8", "qint16", "qint32",
                   "quint8", "quint16", "quint32",
                   "bfloat16", "half"]

def get_attr_strs(attrs):
    attr_strs = []
    for att, val in attrs.iteritems():
        at_str = "Attr(\""
        if isinstance(val, basestring):
            assert(val in attr_data_types)
            at_str = at_str + att + ": " + val + "\")"
        else:
            assert(0)
        # TODO : Handle tensors, shapes, lists, and constraints
        # TODO : Add more semantics
        attr_strs.append(at_str)
    return attr_strs

def get_spec_strs(specs, prefix):
    spec_strs = []
    for spec, val in specs.iteritems():
        spec_str = prefix + "(\""
        if isinstance(val, basestring):
            spec_str = spec_str + spec + ": " + val + "\")"
        else:
            assert(0)
        # TODO : Handle lists
        # TODO : Add more semantics
        spec_strs.append(spec_str)
    return spec_strs

def gen_reg_op_macro_str(op_name, attrs, inputs, outputs):
    macro_name = "REGISTER_OP" + "(\"" + op_name + "\")";
    macro_call = []
    macro_call.append(macro_name)
    macro_call.extend(get_attr_strs(attrs))
    macro_call.extend(get_spec_strs(inputs, "Input"))
    macro_call.extend(get_spec_strs(outputs, "Output"))

    macro_str = "\n.".join(macro_call)
    return macro_str

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
    reg_macro = gen_reg_op_macro_str(op_name, attrs, inputs, outputs)
    contents.append(c.Statement(reg_macro))
    return c.Module(contents)

def generate_op_kernel():
    return

def generate_op_python_wrapper():
    return

def generate_op_gradient():
    return

print(generate_op_registration("ZeroOut", {},
                                       {"to_zero" : "int32"},
                                       {"zeroed": "int32"}))
