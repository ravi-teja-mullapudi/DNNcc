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
    for att, val in attrs:
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
    for spec, val in specs:
        spec_str = prefix + "(\""
        if isinstance(val, basestring):
            spec_str = spec_str + spec + ": " + val + "\")"
        else:
            assert(0)
        # TODO : Handle lists
        # TODO : Add more semantics
        spec_strs.append(spec_str)
    return spec_strs

def gen_reg_op_macro_str(op_name, attrs, inputs, outputs, shape_fn):
    macro_name = "REGISTER_OP" + "(\"" + op_name + "\")";
    macro_call = []
    macro_call.append(macro_name)
    macro_call.extend(get_attr_strs(attrs))
    macro_call.extend(get_spec_strs(inputs, "Input"))
    macro_call.extend(get_spec_strs(outputs, "Output"))
    macro_call.extend(["Shape(" + shape_fn + ")"])
    macro_str = "\n.".join(macro_call)
    return macro_str

def gen_shape_fn(output_shapes):
    shape_fn = []
    shape_fn.append(c.Line(
        "[](::tensorflow::shape_inference::InferenceContext* c)"))

    func_body = []
    for out, dims in output_shapes:
        dim_strs = [ str(d) for d in dims ]
        # dim initializer list
        dim_init_list = "{" + ",".join(dim_strs) + "}"
        # make a shape
        shape_name = "s" + str(out)
        func_body.append(c.Assign("auto " + shape_name,
                                  "c->MakeShape(" + dim_init_list + ")"))
        dim_set_str = "c->set_output(" + str(out) + ", " + shape_name + ")"
        func_body.append(c.Statement(dim_set_str));

    func_body.append(c.Statement("return Status::OK()"));
    body = c.Block(func_body)

    shape_fn.append(body)
    return str(c.Module(shape_fn))

def gen_op_registration(op_name, attrs, inputs, outputs,
                        output_shapes, kernel_class_name):
    # Add the headers into the module
    contents = []
    contents.append(c.Include("tensorflow/core/framework/op.h", system = False))
    contents.append(c.Include("tensorflow/core/framework/shape_inference.h", system = False))

    # Name space declarations
    contents.append(c.Line())
    contents.append(c.Statement("using namespace tensorflow"))
    contents.append(c.Line())

    shape_fn = gen_shape_fn(output_shapes);

    # Registration macro
    reg_macro = gen_reg_op_macro_str(op_name, attrs, inputs, outputs, shape_fn)
    contents.append(c.Statement(reg_macro))
    return c.Module(contents)

def gen_op_kernel_constructor(class_name):
    constructor = []
    const_sign = "explicit %s(OpKernelConstruction* context) : OpKernel(context)" % (class_name)
    constructor.append(c.Line(const_sign))
    constructor.append(c.Block([]))
    return constructor

def gen_op_kernel_compute_fn():
    compute_fn = []
    compute_fn.append(c.Line("void Compute(OpKernelContext* context) override"))
    fn_body = []

    inputs = [c.Comment("Booyah")]
    fn_body.extend(inputs)
    compute_fn.append(c.Block(fn_body))
    return compute_fn

def gen_op_kernel_class_defn(class_name):
    class_defn = []
    class_defn.append(c.Line("class " + class_name + ": public OpKernel"))

    class_body = []
    class_body.append(c.Line("public:"))

    class_construct = gen_op_kernel_constructor(class_name)
    class_body.extend(class_construct)

    class_body.append(c.Line())
    compute_fn = gen_op_kernel_compute_fn()
    class_body.extend(compute_fn)

    body = c.Block(class_body)
    class_defn.append(body)
    # TODO: fix this at some point
    class_defn.append(c.Line(";"))
    return class_defn

def gen_op_kernel(kernel_class_name):
    contents = []
    contents.append(c.Include("tensorflow/core/framework/op_kernel.h", system = False))

    # Name space declarations
    contents.append(c.Line())
    contents.append(c.Statement("using namespace tensorflow"))
    contents.append(c.Line())

    class_defn = gen_op_kernel_class_defn(kernel_class_name)

    contents.extend(class_defn)
    return c.Module(contents)

def gen_op_python_wrapper():
    return

def gen_op_gradient():
    return

print(gen_op_registration("ZeroOut",
                          {},
                          [("to_zero", "int32")],
                          [("zeroed", "int32")],
                          [(0, [1000])],
                          "ZeroOutOp"
                          ))

print(gen_op_kernel("ZeroOutOp"))
