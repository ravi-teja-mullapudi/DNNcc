CXX ?= g++
CXXFLAGS += -O2 -g -Wall -std=c++11 -rdynamic

TF_INC = /usr/local/lib/python2.7/dist-packages/tensorflow/include
INC=-I${TF_INC}

HALIDE_PATH = /home/ravi/Halide
HALIDE_INC += -I$(HALIDE_PATH)/include -I$(HALIDE_PATH)/tools
HALIDE_LIB += -L$(HALIDE_PATH)/bin -lHalide

all: layer_norm_fused_op lnorm_gen lnorm_op.so

layer_norm_fused_op: register_ops.cc layer_norm_fused_op.h layer_norm_fused_op.cc layer_norm_fused_grad_op.cc layer_norm_fused_op_gpu.cu.cc
	nvcc -std=c++11 -c -o layer_norm_fused_op_gpu.cu.o layer_norm_fused_op_gpu.cu.cc \
	$(INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -arch=sm_61
	g++ -std=c++11 -shared -o layer_norm_fused_op.so register_ops.cc layer_norm_fused_op.h \
	layer_norm_fused_grad_op.cc layer_norm_fused_op.cc layer_norm_fused_op_gpu.cu.o \
	$(INC) -L /usr/local/cuda/lib64/ -fPIC -lcudart -O2 -DNDEBUG

lnorm_gen: halide_layer_norm.cpp
	$(CXX) $(CXXFLAGS) halide_layer_norm.cpp -o lnorm_gen \
		${HALIDE_LIB} ${HALIDE_INC}
	./lnorm_gen

lnorm_op.so: lnorm_gen layer_norm_op.cpp ../op_interface_gen/tf_halide_utils.h
	$(CXX) -O2 -std=c++11 -shared layer_norm_op.cpp gen_halide_ln_2d.a gen_halide_ln_4d.a \
			-o lnorm_op.so -fPIC -I $(TF_INC) -I ../op_interface_gen/

clean:
	rm -rf layer_norm_fused_op.so layer_norm_fused_op_gpu.cu.o lnorm_gen \
		   gen_halide_ln_2d.a gen_halide_ln_2d.h gen_halide_ln_4d.a gen_halide_ln_4d.h \
		   lnorm_op.so
