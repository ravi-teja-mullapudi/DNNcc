#pragma once

#include <memory>
#include <vector>
#include <cassert>
#include "UtilsCUDA.h"

// TODO
// 1) Handle multiple devices
// 2) Abstract different device API's currently everything is tied to CUDA
// 3) Write more tests

template <class T>
struct NDArray {
    std::shared_ptr<T> host_alloc;
    std::shared_ptr<T> dev_alloc;
    std::vector<int> strides;
    size_t buf_size;

    struct delete_cuda_ptr {
        void operator()(T *p) const {
            cudaFree(p);
        }
    };

    NDArray(std::vector<int> _strides) : strides(_strides) {
        if (strides.size() >= 1) {
            buf_size = 1;
            for (auto& s: strides) {
                buf_size *= s;
            }
            T* host_ptr = new T[buf_size];
            host_alloc.reset(host_ptr);
        } else {
            buf_size = 0;
        }
    }

    void initialize(T val) {
        T* host_ptr = host_alloc.get();
        for (size_t i = 0; i < buf_size; i++) {
            host_ptr[i] = val;
        }
    }

    template <typename F>
    void initialize(F& op) {
        T* host_ptr = host_alloc.get();
        for (size_t i = 0; i < buf_size; i++) {
            host_ptr[i] = op();
        }
    }

    inline T& operator()(int d1) {
        assert(strides.size() == 1);
        return host_alloc.get()[d1];
    }

    inline T& operator()(int d1, int d2) {
        assert(strides.size() == 2);
        int offset = d1 * strides[1] + d2;
        return host_alloc.get()[offset];
    }

    inline T& operator()(int d1, int d2, int d3) {
        assert(strides.size() == 3);
        int offset = d1 * strides[1] * strides[2] + d2 * strides[2] + d3;
        return host_alloc.get()[offset];
    }

    inline T& operator()(int d1, int d2, int d3, int d4) {
        assert(strides.size() == 4);
        int offset = d1 * strides[1] * strides[2] * strides[3] +
                     d2 * strides[2] * strides[3] + d3 * strides[3] + d4;
        return host_alloc.get()[offset];
    }

    inline const T& operator()(int d1) const {
        assert(strides.size() == 1);
        return host_alloc.get()[d1];
    }

    inline const T& operator()(int d1, int d2) const {
        assert(strides.size() == 2);
        int offset = d1 * strides[1] + d2;
        return host_alloc.get()[offset];
    }

    inline const T& operator()(int d1, int d2, int d3) const {
        assert(strides.size() == 3);
        int offset = d1 * strides[1] * strides[2] + d2 * strides[2] + d3;
        return host_alloc.get()[offset];
    }

    inline const T& operator()(int d1, int d2, int d3, int d4) const {
        assert(strides.size() == 4);
        int offset = d1 * strides[1] * strides[2] * strides[3] +
                     d2 * strides[2] * strides[3] + d3 * strides[3] + d4;
        return host_alloc.get()[offset];
    }

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
