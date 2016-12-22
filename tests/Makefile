NVCC ?= nvcc
NVCCFLAGS += -g -std=c++11
CUDNNLDFLAGS += -lcudnn -L/usr/local/cuda/lib64

all: ndarray_test conv2d_cuda_test

ndarray_test: NDArrayTest.cu ../NDArray.h ../UtilsCUDA.h
	$(NVCC) $(NVCCFLAGS) NDArrayTest.cu -I../ -o ndarray_test

conv2d_cuda_test: Conv2dTest.cu ../NDArray.h ../Utils.h ../UtilsCUDNN.h
	$(NVCC) $(NVCCFLAGS) Conv2dTest.cu $(CUDNNLDFLAGS) -I../ -o conv2d_cuda_test

clean:
	rm -rf ndarray_test conv2d_cuda_test
