#pragma once

#include <memory>
#include <vector>
#include <cassert>
#include "boost/variant.hpp"

// TODO
// 1) Handle multiple devices
// 2) Write more tests

template <class T>
class NDArray {
    public:
    std::shared_ptr<T> host_alloc;
    std::vector<int> dim_sizes;
    size_t buf_size;

    NDArray() {
        buf_size = 0;
    }

    NDArray(std::vector<int> _dim_sizes) : dim_sizes(_dim_sizes) {
        if (dim_sizes.size() >= 1) {
            buf_size = 1;
            for (auto& s: dim_sizes) {
                buf_size *= s;
            }
            T* host_ptr = new T[buf_size];
            host_alloc.reset(host_ptr);
        } else {
            buf_size = 0;
        }
    }

    int dimensions() { return dim_sizes.size(); }

    int extent(int dim_id) {
        assert(dim_id < (int)dim_sizes.size());
        return dim_sizes[dim_id];
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
        assert(dim_sizes.size() == 1);
        return host_alloc.get()[d1];
    }

    inline T& operator()(int d1, int d2) {
        assert(dim_sizes.size() == 2);
        int offset = d1 * dim_sizes[1] + d2;
        return host_alloc.get()[offset];
    }

    inline T& operator()(int d1, int d2, int d3) {
        assert(dim_sizes.size() == 3);
        int offset = d1 * dim_sizes[1] * dim_sizes[2] + d2 * dim_sizes[2] + d3;
        return host_alloc.get()[offset];
    }

    inline T& operator()(int d1, int d2, int d3, int d4) {
        assert(dim_sizes.size() == 4);
        int offset = d1 * dim_sizes[1] * dim_sizes[2] * dim_sizes[3] +
                     d2 * dim_sizes[2] * dim_sizes[3] + d3 * dim_sizes[3] + d4;
        return host_alloc.get()[offset];
    }

    inline const T& operator()(int d1) const {
        assert(dim_sizes.size() == 1);
        return host_alloc.get()[d1];
    }

    inline const T& operator()(int d1, int d2) const {
        assert(dim_sizes.size() == 2);
        int offset = d1 * dim_sizes[1] + d2;
        return host_alloc.get()[offset];
    }

    inline const T& operator()(int d1, int d2, int d3) const {
        assert(dim_sizes.size() == 3);
        int offset = d1 * dim_sizes[1] * dim_sizes[2] + d2 * dim_sizes[2] + d3;
        return host_alloc.get()[offset];
    }

    inline const T& operator()(int d1, int d2, int d3, int d4) const {
        assert(dim_sizes.size() == 4);
        int offset = d1 * dim_sizes[1] * dim_sizes[2] * dim_sizes[3] +
                     d2 * dim_sizes[2] * dim_sizes[3] + d3 * dim_sizes[3] + d4;
        return host_alloc.get()[offset];
    }

    void copy(const NDArray<T>& other) {
        for (size_t d = 0; d < dim_sizes.size(); d++) {
            assert(dim_sizes[d] == other.dim_sizes[d]);
        }
        T* host_ptr = host_alloc.get();
        for (size_t i = 0; i < buf_size; i++) {
            host_ptr[i] = other.host_alloc.get()[i];
        }
    }

    /* Simple inplace arithmetic operations. */
    void add(const NDArray<T>& other) {
        for (size_t d = 0; d < dim_sizes.size(); d++) {
            assert(dim_sizes[d] == other.dim_sizes[d]);
        }
        T* host_ptr = host_alloc.get();
        for (size_t i = 0; i < buf_size; i++) {
            host_ptr[i] += other.host_alloc.get()[i];
        }
    }
};

using NDArray_t = boost::variant<NDArray<double>,
                                 NDArray<float>,
                                 NDArray<int64_t>,
                                 NDArray<int32_t>,
                                 NDArray<int16_t>,
                                 NDArray<int8_t>,
                                 NDArray<uint64_t>,
                                 NDArray<uint32_t>,
                                 NDArray<uint16_t>,
                                 NDArray<uint8_t>>;

template <class T>
NDArray<T>& get_ndarray(NDArray_t& arr) {
    return boost::get<NDArray<T>>(arr);
}

enum DataType {
    Float64,
    Float32,
    Int64,
    Int32,
    Int16,
    Int8,
    UInt64,
    UInt32,
    UInt16,
    UInt8
};
