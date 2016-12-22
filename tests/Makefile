CXX ?= g++
CXXFLAGS += -g -Wall -std=c++11 -fopenmp -rdynamic

NVCC ?= nvcc
NVCCFLAGS += -g -std=c++11

CUDNNLDFLAGS += -lcudnn -L/usr/local/cuda/lib64

all: ndarray_cuda_test op_ref.o conv2d_cuda_test

ndarray_cuda_test: NDArrayTest.cu ../NDArrayCUDA.h ../UtilsCUDA.h
	$(NVCC) $(NVCCFLAGS) NDArrayTest.cu -I../ -o ndarray_cuda_test

op_ref.o: ../NDArray.h ../OpRef.h ../OpRef.cpp
	$(CXX) $(CXXFLAGS) ../OpRef.cpp -I../ -c -o op_ref.o

conv2d_cuda_test: Conv2dTest.cu ../NDArray.h ../Utils.h ../UtilsCUDNN.h op_ref.o
	$(NVCC) $(NVCCFLAGS) Conv2dTest.cu $(CUDNNLDFLAGS) -I../ -o conv2d_cuda_test

clean:
	rm -rf ndarray_cuda_test conv2d_cuda_test
