#pragma once

#include "NDArray.h"
#include "UtilsCUDA.h"

// TODO
// 1) Handle multiple devices
template <class T>
class NDArrayCUDA: public NDArray<T> {
    public:
    using NDArray<T>::host_alloc;
    using NDArray<T>::buf_size;

    std::shared_ptr<T> dev_alloc;
    struct delete_cuda_ptr {
        void operator()(T *p) const {
            cudaFree(p);
        }
    };

    NDArrayCUDA(std::vector<int> _strides) : NDArray<T>(_strides) {}

    void copy_to_device() {
        assert(host_alloc && dev_alloc);
        if (dev_alloc) {
            gpu_err_chk(cudaMemcpy(dev_alloc.get(),
                        host_alloc.get(), sizeof(T) * buf_size,
                        cudaMemcpyHostToDevice));
        }
    }

    void copy_from_device() {
        assert(host_alloc && dev_alloc);
        if (dev_alloc) {
            gpu_err_chk(cudaMemcpy(host_alloc.get(),
                        dev_alloc.get(), sizeof(T) * buf_size,
                        cudaMemcpyDeviceToHost));
        }
    }

    void device_allocate() {
        assert(host_alloc);
        if (buf_size > 0) {
            T* dev_ptr;
            gpu_err_chk(cudaMalloc((void **)&dev_ptr, sizeof(T) * buf_size));
            dev_alloc.reset(dev_ptr, delete_cuda_ptr());
        }
    }
};
