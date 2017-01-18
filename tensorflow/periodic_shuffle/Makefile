CXX ?= g++
CXXFLAGS += -O2 -g -Wall -std=c++11 -rdynamic

TF_INC = /usr/local/lib/python2.7/dist-packages/tensorflow/include
INC=-I${TF_INC}

HALIDE_PATH = /home/ravi/Halide
HALIDE_INC += -I$(HALIDE_PATH)/include -I$(HALIDE_PATH)/tools
HALIDE_LIB += -L$(HALIDE_PATH)/bin -lHalide

all: periodic_shuffle_gen periodic_shuffle_op.so

periodic_shuffle_gen: halide_periodic_shuffle.cpp
	$(CXX) $(CXXFLAGS) halide_periodic_shuffle.cpp -o periodic_shuffle_gen \
		${HALIDE_LIB} ${HALIDE_INC}
	./periodic_shuffle_gen

periodic_shuffle_op.so: periodic_shuffle_gen periodic_shuffle_op.cpp ../op_interface_gen/tf_halide_utils.h
	$(CXX) -O2 -std=c++11 -shared periodic_shuffle_op.cpp gen_periodic_shuffle.a\
			-o periodic_shuffle_op.so -fPIC -I $(TF_INC) -I ../op_interface_gen/

clean:
	rm -rf periodic_shuffle_op.so periodic_shuffle_gen \
		   gen_periodic_shuffle.a gen_periodic_shuffle.h
