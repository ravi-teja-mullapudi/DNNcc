CXX ?= g++
CXXFLAGS += -O2 -g -Wall -std=c++11 -rdynamic

TF_INC = /usr/local/lib/python2.7/dist-packages/tensorflow/include

HALIDE_PATH = /home/ravi/Halide
HALIDE_INC += -I$(HALIDE_PATH)/include -I$(HALIDE_PATH)/tools
HALIDE_LIB += -L$(HALIDE_PATH)/bin -lHalide

bnorm_gen: halide_bnorm.cpp
	$(CXX) $(CXXFLAGS) halide_bnorm.cpp -o bnorm_gen \
		${HALIDE_LIB} ${HALIDE_INC}
	./bnorm_gen

bnorm_op.so: bnorm_gen bnorm_op.cpp
	$(CXX) -O2 -std=c++11 -shared bnorm_op.cpp gen_halide_bn_inference.a -o bnorm_op.so -fPIC -I $(TF_INC)

clean:
	rm -rf bnorm_gen gen_halide_bn_inference.h gen_halide_bn_inference.a bnorm_op.so
