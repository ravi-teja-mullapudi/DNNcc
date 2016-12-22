NVCC ?= nvcc
NVCCFLAGS += -g -std=c++11

all: ndarray_test

ndarray_test: NDArrayTest.cu ../NDArray.h ../UtilsCUDA.h
	$(NVCC) $(NVCCFLAGS) NDArrayTest.cu -I../ -o ndarray_test

clean:
	rm -rf ndarray_test
