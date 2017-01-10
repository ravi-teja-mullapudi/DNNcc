CXX ?= g++
CXXFLAGS += -O2 -g -Wall -std=c++11 -rdynamic

TF_INC = /usr/local/lib/python2.7/dist-packages/tensorflow/include

HALIDE_PATH = /home/ravi/Halide
HALIDE_INC += -I$(HALIDE_PATH)/include -I$(HALIDE_PATH)/tools
HALIDE_LIB += -L$(HALIDE_PATH)/bin -lHalide

zero_op_gen: halide_zero_op.cpp
	$(CXX) $(CXXFLAGS) halide_zero_op.cpp -o zero_op_gen \
		${HALIDE_LIB} ${HALIDE_INC}
	./zero_op_gen

zero_op.so: zero_op_gen zero_op.cpp
	$(CXX) -O2 -std=c++11 -shared zero_op.cpp gen_halide_zero_op.a -o zero_op.so -fPIC -I $(TF_INC)

clean:
	rm -rf zero_op_gen gen_halide_zero_op.a gen_halide_zero_op.h zero_op.so
