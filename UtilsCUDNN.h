#pragma once

#include <cudnn.h>
#include <vector>
#include <cstdio>
#include <memory>
#include "NDArray.h"

#define cudnn_status_chk(ans) { cudnn_assert((ans), __FILE__, __LINE__); }
void cudnn_assert(cudnnStatus_t status, const char* file, int line, bool abort = true) {
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr,"CUDNNassert: %s %s %d\n",
                cudnnGetErrorString(status), file, line);
        if(abort) {
            exit(status);
        }
    }
}

class CudnnHandle {
    std::shared_ptr<cudnnHandle_t> handle_;

    struct CudnnHandleDeleter {
        void operator()(cudnnHandle_t* handle) {
            cudnnDestroy(*handle);
            delete handle;
        }
    };

    public:
    CudnnHandle() : handle_(new cudnnHandle_t, CudnnHandleDeleter()) {
        cudnn_status_chk(cudnnCreate(handle_.get()));
    }

    cudnnHandle_t handle() const { return *handle_; };
};

template<typename T>
class FilterDescriptorNd {
    std::shared_ptr<cudnnFilterDescriptor_t> desc_;

    struct FilterDescriptorNdDeleter {
        void operator()(cudnnFilterDescriptor_t * desc) {
            cudnnDestroyFilterDescriptor(*desc);
            delete desc;
        }
    };

    public:
    FilterDescriptorNd() {}

    FilterDescriptorNd(const cudnnTensorFormat_t tensor_format,
                       const std::vector<int>& dims) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value) {
            type = CUDNN_DATA_FLOAT;
        } else {
            throw std::runtime_error("Unknown type");
        }

        cudnnFilterDescriptor_t * desc = new cudnnFilterDescriptor_t;
        cudnn_status_chk(cudnnCreateFilterDescriptor(desc));
        cudnn_status_chk(cudnnSetFilterNdDescriptor(*desc,
                                                      type,
                                                      tensor_format,
                                                      dims.size(),
                                                      &dims[0]));

        desc_.reset(desc, FilterDescriptorNdDeleter());
    }
    cudnnFilterDescriptor_t desc() { return *desc_; }
};

template<typename T>
class TensorDescriptorNd {
    std::shared_ptr<cudnnTensorDescriptor_t> desc_;

    struct TensorDescriptorNdDeleter {
        void operator()(cudnnTensorDescriptor_t* desc) {
            cudnnDestroyTensorDescriptor(*desc);
            delete desc;
        }
    };

    public:
    TensorDescriptorNd(const std::vector<int>& dim,
                       const std::vector<int>& stride) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value) {
            type = CUDNN_DATA_FLOAT;
        } else {
            std::cerr << "Unknown type" << std::endl;
            exit(-1);
        }

        cudnnTensorDescriptor_t* desc = new cudnnTensorDescriptor_t;

        cudnn_status_chk(cudnnCreateTensorDescriptor(desc));
        cudnn_status_chk(cudnnSetTensorNdDescriptor(*desc, type, dim.size(),
                                                    &dim[0], &stride[0]));

        desc_.reset(desc, TensorDescriptorNdDeleter());
    }

    cudnnTensorDescriptor_t desc() const { return *desc_; }
};

template<typename T>
class TensorDescriptorNdArray {
    cudnnTensorDescriptor_t* desc_;

    /*struct TensorDescriptorNdDeleter {
        void operator()(cudnnTensorDescriptor_t* descArray_) {
			for(int i =0; i < size; i++) {
            	cudnnDestroyTensorDescriptor(descArray[]);
            delete desc;
        }
    };*/

    public:
    TensorDescriptorNdArray(const std::vector<int>& dim,
                            const std::vector<int>& stride,
                            int array_size) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value) {
            type = CUDNN_DATA_FLOAT;
        } else {
            std::cerr << "Unknown type" << std::endl;
            exit(-1);
        }

        cudnnTensorDescriptor_t* desc_ = new cudnnTensorDescriptor_t[array_size];

        for(int i = 0; i < array_size; i++) {
            cudnn_status_chk(cudnnCreateTensorDescriptor(&desc_[i]));
            cudnn_status_chk(cudnnSetTensorNdDescriptor(desc_[i], type, dim.size(),
                                                        &dim[0], &stride[0]));
        }
    }

    cudnnTensorDescriptor_t* desc() const { return desc_; }
    cudnnTensorDescriptor_t desc_elem() const { return desc_[0]; }
};

template<typename T>
class TensorDescriptor4d {
    std::shared_ptr<cudnnTensorDescriptor_t> desc_;

    struct TensorDescriptor4dDeleter {
        void operator()(cudnnTensorDescriptor_t* desc) {
            cudnnDestroyTensorDescriptor(*desc);
            delete desc;
        }
    };

    public:
    TensorDescriptor4d() {}
    TensorDescriptor4d(const cudnnTensorFormat_t tensor_format,
                       const int n, const int c, const int h, const int w) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value) {
            type = CUDNN_DATA_FLOAT;
        } else {
            std::cerr << "Unknown type" << std::endl;
            exit(-1);
        }

        cudnnTensorDescriptor_t* desc = new cudnnTensorDescriptor_t;
        cudnn_status_chk(cudnnCreateTensorDescriptor(desc));
        cudnn_status_chk(cudnnSetTensor4dDescriptor(*desc,
                                                     tensor_format,
                                                     type,
                                                     n,
                                                     c,
                                                     h,
                                                     w));

        desc_.reset(desc, TensorDescriptor4dDeleter());
    }

    cudnnTensorDescriptor_t desc() const { return *desc_; }
};

template<typename T>
class FilterDescriptor4d {
    std::shared_ptr<cudnnFilterDescriptor_t> desc_;

    struct FilterDescriptor4dDeleter {
        void operator()(cudnnFilterDescriptor_t* desc) {
            cudnnDestroyFilterDescriptor(*desc);
            delete desc;
        }
    };

    public:
    FilterDescriptor4d(const cudnnTensorFormat_t tensor_format,
                       int k, int c, int h, int w) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value) {
            type = CUDNN_DATA_FLOAT;
        } else {
            std::cerr << "Unknown type" << std::endl;
            exit(-1);
        }

        cudnnFilterDescriptor_t * desc = new cudnnFilterDescriptor_t;
        cudnn_status_chk(cudnnCreateFilterDescriptor(desc));
        cudnn_status_chk(cudnnSetFilter4dDescriptor(*desc,
                                                     type,
                                                     tensor_format,
                                                     k, c, h, w));

        desc_.reset(desc, FilterDescriptor4dDeleter());
    }

    cudnnFilterDescriptor_t desc() const { return *desc_; }

};

class Pooling3dDescriptor {
    std::shared_ptr<cudnnPoolingDescriptor_t> desc_;

    struct Pooling3dDescriptorDeleter {
        void operator()(cudnnPoolingDescriptor_t * desc) {
            cudnnDestroyPoolingDescriptor(*desc);
            delete desc;
        }
    };

    public:
    Pooling3dDescriptor(cudnnPoolingMode_t mode, int k_ch, int k_h, int k_w,
                        int pad_ch, int pad_h, int pad_w,
                        int stride_ch, int stride_h, int stride_w) :

        desc_(new cudnnPoolingDescriptor_t, Pooling3dDescriptorDeleter()) {

        cudnn_status_chk(cudnnCreatePoolingDescriptor(desc_.get()));
        int window[3] = {k_ch, k_h, k_w};
        int padding[3] = {pad_ch, pad_h, pad_w};
        int stride[3] = {stride_ch, stride_h, stride_w};
        cudnn_status_chk(cudnnSetPoolingNdDescriptor(*desc_,
                                                     mode,
                                                     // What in the world is this?
                                                     CUDNN_NOT_PROPAGATE_NAN,
                                                     3,
                                                     window,
                                                     padding,
                                                     stride));
    }

    cudnnPoolingDescriptor_t desc() const { return *desc_; };
};

class Pooling2dDescriptor {
    std::shared_ptr<cudnnPoolingDescriptor_t> desc_;

    struct Pooling2dDescriptorDeleter {
        void operator()(cudnnPoolingDescriptor_t * desc) {
            cudnnDestroyPoolingDescriptor(*desc);
            delete desc;
        }
    };

    public:
    Pooling2dDescriptor(cudnnPoolingMode_t mode, int k_h, int k_w,
                        int pad_h, int pad_w, int stride_h, int stride_w) :
        desc_(new cudnnPoolingDescriptor_t, Pooling2dDescriptorDeleter()) {

        cudnn_status_chk(cudnnCreatePoolingDescriptor(desc_.get()));
        cudnn_status_chk(cudnnSetPooling2dDescriptor(*desc_,
                                                     mode,
                                                     CUDNN_NOT_PROPAGATE_NAN,
                                                     k_h,
                                                     k_w,
                                                     pad_h,
                                                     pad_w,
                                                     stride_h,
                                                     stride_w));
    }

    cudnnPoolingDescriptor_t desc() const { return *desc_; };
};

class LRNDescriptor {
    std::shared_ptr<cudnnLRNDescriptor_t> desc_;

    struct LRNDescriptorDeleter {
        void operator()(cudnnLRNDescriptor_t * desc) {
            cudnnDestroyLRNDescriptor(*desc);
            delete desc;
        }
    };

    public:
    LRNDescriptor(unsigned radius, double alpha, double beta, double bias) :
        desc_(new cudnnLRNDescriptor_t, LRNDescriptorDeleter()) {

            cudnn_status_chk(cudnnCreateLRNDescriptor(desc_.get()));
            cudnn_status_chk(cudnnSetLRNDescriptor(*desc_,
                                                   (radius * 2)+1,
                                                   alpha,
                                                   beta,
                                                   2));
    }

    cudnnLRNDescriptor_t desc() const { return *desc_; };

};

class ConvolutionDescriptor {
    std::shared_ptr<cudnnConvolutionDescriptor_t> desc_;

    struct ConvolutionDescriptorDeleter {
        void operator()(cudnnConvolutionDescriptor_t * desc) {
            cudnnDestroyConvolutionDescriptor(*desc);
            delete desc;
        }
    };

    public:
    ConvolutionDescriptor(int pad_h, int pad_w, int hstride, int wstride) :
        desc_(new cudnnConvolutionDescriptor_t, ConvolutionDescriptorDeleter()) {

        cudnn_status_chk(cudnnCreateConvolutionDescriptor(desc_.get()));
        cudnn_status_chk(cudnnSetConvolution2dDescriptor(*desc_,
                                                         pad_h,
                                                         pad_w,
                                                         hstride,
                                                         wstride,
                                                         1,
                                                         1,
                                                         CUDNN_CROSS_CORRELATION));
    }

    cudnnConvolutionDescriptor_t desc() const { return *desc_; };
};

template <typename T>
class RNNDescriptor {
    std::shared_ptr<cudnnRNNDescriptor_t> desc_;

    struct RNNDescriptorDeleter {
        void operator()(cudnnRNNDescriptor_t* rnn_desc) {
            cudnnDestroyRNNDescriptor(*rnn_desc);
            delete rnn_desc;
        }
    };

    public:
    RNNDescriptor() {}
    RNNDescriptor(int hidden_size, int num_layers,
                  cudnnDropoutDescriptor_t dropout_desc,
                  cudnnRNNInputMode_t input_mode,
                  cudnnDirectionMode_t direction,
                  std::string rnn_type) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value) {
            type = CUDNN_DATA_FLOAT;
        } else {
            std::cerr << "Unknown type" << std::endl;
            exit(-1);
        }

        cudnnRNNMode_t rnn_mode;
        if (rnn_type == "vanilla") {
            rnn_mode = CUDNN_RNN_RELU;
        } else if (rnn_type == "gru") {
            rnn_mode = CUDNN_GRU;
        } else if (rnn_type == "lstm") {
            rnn_mode = CUDNN_LSTM;
        } else {
            std::cerr << "Unknow rnn type" << std::endl;
            exit(-1);
        }

        cudnnRNNDescriptor_t* desc = new cudnnRNNDescriptor_t;

        cudnn_status_chk(cudnnCreateRNNDescriptor(desc));
        cudnn_status_chk(cudnnSetRNNDescriptor(*desc,
                                               hidden_size,
                                               num_layers,
                                               dropout_desc,
                                               input_mode,
                                               direction,
                                               rnn_mode,
                                               type));

        desc_.reset(desc, RNNDescriptorDeleter());
    }

    cudnnRNNDescriptor_t desc() { return *desc_; }
};

class CudnnLRN {

    TensorDescriptor4d<float> input_desc_;
    TensorDescriptor4d<float> output_desc_;

    std::vector<int> output_dims_;

    cudnnConvolutionFwdAlgo_t fwd_algo_;
    LRNDescriptor lrn_desc_;
    CudnnHandle cudnn_handle_;

    public:
    CudnnLRN(int w, int h, int c, int n,
             unsigned int radius, double alpha, double beta, double bias) :
             cudnn_handle_(),
             input_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
             lrn_desc_(radius, alpha, beta, bias) {

        output_desc_ = TensorDescriptor4d<float>(CUDNN_TENSOR_NCHW, n, c, h, w);
    }

    void forward(NDArray<float>& input, NDArray<float>& output) {
        float alpha = 1.0f;
        float beta = 0.0f;
        cudnn_status_chk(cudnnLRNCrossChannelForward(cudnn_handle_.handle(),
                                                     lrn_desc_.desc(),
                                                     CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                                     &alpha,
                                                     input_desc_.desc(),
                                                     input.dev_alloc.get(),
                                                     &beta,
                                                     output_desc_.desc(),
                                                     output.dev_alloc.get()));
    }
};

class CudnnAvgpool2d {

    TensorDescriptor4d<float> input_desc_;
    TensorDescriptor4d<float> output_desc_;

    std::vector<int> output_dims_;

    Pooling2dDescriptor pool_desc_;
    CudnnHandle cudnn_handle_;

    const float alpha_ = 1.0f;
    const float beta_  = 0.0f;

    public:
    CudnnAvgpool2d(int w, int h, int c, int n, int k_w, int k_h,
                   int pad_w, int pad_h, int stride_w, int stride_h) :
                   cudnn_handle_(),
                   input_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
                   pool_desc_(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                              k_h, k_w, pad_h, pad_w, stride_h, stride_w) {

        int out_h, out_w, out_c, out_n;
        // Get output dimensions
        cudnn_status_chk(cudnnGetPooling2dForwardOutputDim(pool_desc_.desc(),
                                                           input_desc_.desc(),
                                                           &out_n,
                                                           &out_c,
                                                           &out_h,
                                                           &out_w));

        output_desc_ = TensorDescriptor4d<float>(CUDNN_TENSOR_NCHW,
                                                 out_n, out_c, out_h, out_w);

        output_dims_ = {out_w, out_h, out_c, out_n};
    }

    std::vector<int> get_output_dims() { return output_dims_; }

    void forward(NDArray<float>& input, NDArray<float>& output) {
        cudnn_status_chk(cudnnPoolingForward(cudnn_handle_.handle(),
                                             pool_desc_.desc(),
                                             &alpha_,
                                             input_desc_.desc(),
                                             input.dev_alloc.get(),
                                             &beta_,
                                             output_desc_.desc(),
                                             output.dev_alloc.get()));
    }
};

class CudnnMaxpool2d {

    TensorDescriptor4d<float> input_desc_;
    TensorDescriptor4d<float> output_desc_;

    std::vector<int> output_dims_;

    Pooling2dDescriptor pool_desc_;
    CudnnHandle cudnn_handle_;

    const float alpha_ = 1.f;
    const float beta_  = 0.f;

    public:
    CudnnMaxpool2d(int w, int h, int c, int n, int k_w, int k_h,
                   int pad_w, int pad_h, int stride_w, int stride_h) :
                   cudnn_handle_(),
                   input_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
                   pool_desc_(CUDNN_POOLING_MAX, k_h, k_w,
                              pad_h, pad_w, stride_h, stride_w) {

        int out_h, out_w, out_c, out_n;
        // Get output dimensions
        cudnn_status_chk(cudnnGetPooling2dForwardOutputDim(pool_desc_.desc(),
                                                           input_desc_.desc(),
                                                           &out_n,
                                                           &out_c,
                                                           &out_h,
                                                           &out_w));

        output_desc_ = TensorDescriptor4d<float>(CUDNN_TENSOR_NCHW,
                                                 out_n, out_c, out_h, out_w);

        output_dims_ = {out_w, out_h, out_c, out_n};
    }

    std::vector<int> get_output_dims() { return output_dims_; }

    void forward(NDArray<float>& input, NDArray<float>& output) {
        cudnn_status_chk(cudnnPoolingForward(cudnn_handle_.handle(),
                                             pool_desc_.desc(),
                                             &alpha_,
                                             input_desc_.desc(),
                                             input.dev_alloc.get(),
                                             &beta_,
                                             output_desc_.desc(),
                                             output.dev_alloc.get()));
    }
};

class CudnnConv2d {

    TensorDescriptor4d<float> input_desc_;
    TensorDescriptor4d<float> output_desc_;

    FilterDescriptor4d<float> weight_desc_;

    std::vector<int> output_dims_;

    cudnnConvolutionFwdAlgo_t fwd_algo_;
    ConvolutionDescriptor conv_desc_;
    CudnnHandle cudnn_handle_;

    const float alpha_ = 1.f;
    const float beta_  = 0.f;

    size_t fwd_workspace_size_;
    NDArray<float> fwd_workspace_;

    public:
    CudnnConv2d(int w, int h, int c, int n, int k, int r, int s,
                int pad_w, int pad_h, int wstride, int hstride) :
                cudnn_handle_(),
                input_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
                weight_desc_(CUDNN_TENSOR_NCHW, k, c, r, s),
                conv_desc_(pad_h, pad_w, hstride, wstride),
                fwd_workspace_({}) {

        int out_h, out_w, out_c, out_n;
        // Get output dimensions
        cudnn_status_chk(cudnnGetConvolution2dForwardOutputDim(conv_desc_.desc(),
                                                               input_desc_.desc(),
                                                               weight_desc_.desc(),
                                                               &out_n,
                                                               &out_c,
                                                               &out_h,
                                                               &out_w));

        output_desc_ = TensorDescriptor4d<float>(CUDNN_TENSOR_NCHW,
                                                 out_n, out_c, out_h, out_w);

        output_dims_ = {out_w, out_h, out_c, out_n};

        // Pick forward convolution algorithm
        cudnn_status_chk(cudnnGetConvolutionForwardAlgorithm(cudnn_handle_.handle(),
                                                             input_desc_.desc(),
                                                             weight_desc_.desc(),
                                                             conv_desc_.desc(),
                                                             output_desc_.desc(),
                                                             CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                             0,
                                                             &fwd_algo_));

        // Set fwd workspace size
        cudnn_status_chk(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_.handle(),
                                                                 input_desc_.desc(),
                                                                 weight_desc_.desc(),
                                                                 conv_desc_.desc(),
                                                                 output_desc_.desc(),
                                                                 fwd_algo_,
                                                                 &fwd_workspace_size_));

        fwd_workspace_ =
            NDArray<float>({static_cast<int>(fwd_workspace_size_ / sizeof(float))});
        fwd_workspace_.device_allocate();
    }

    std::vector<int> get_output_dims() { return output_dims_; }

    std::string get_fwd_algo_string() {
        if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) {
            return "IMPLICIT_GEMM";
        } else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
            return "IMPLICIT_PRECOMP_GEMM";
        } else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_GEMM) {
            return "GEMM";
        } else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_DIRECT) {
            return "DIRECT";
        } else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_FFT) {
            return "FFT";
        } else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) {
            return "FFT_TILING";
        } else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD) {
            return "WINOGRAD";
        } else {
            std::cerr << "Illegal algorithm passed to get_fwd_algo_string. Algo: " << fwd_algo_ << std::endl;
            exit(-1);
        }
    }

    void forward(NDArray<float>& input, NDArray<float>& filter, NDArray<float>& output) {
        // Convolution forward.
        cudnn_status_chk(cudnnConvolutionForward(cudnn_handle_.handle(),
                                                 &alpha_,
                                                 input_desc_.desc(),
                                                 input.dev_alloc.get(),
                                                 weight_desc_.desc(),
                                                 filter.dev_alloc.get(),
                                                 conv_desc_.desc(),
                                                 fwd_algo_,
                                                 fwd_workspace_.dev_alloc.get(),
                                                 fwd_workspace_size_,
                                                 &beta_,
                                                 output_desc_.desc(),
                                                 output.dev_alloc.get()));
    }
};


class CudnnLSTM {

    TensorDescriptorNdArray<float> input_desc_;
    TensorDescriptorNdArray<float> output_desc_;
    TensorDescriptorNd<float> h_init_desc_;
    TensorDescriptorNd<float> c_init_desc_;
    TensorDescriptorNd<float> h_final_desc_;
    TensorDescriptorNd<float> c_final_desc_;
    FilterDescriptorNd<float> weight_desc_;

	cudnnDropoutDescriptor_t *dropout_desc_;
    RNNDescriptor<float> rnn_desc_;
    CudnnHandle cudnn_handle_;

    const float alpha_ = 1.f;
    const float beta_  = 0.f;

    size_t fwd_workspace_size_;
    NDArray<float> fwd_workspace_;

    public:
    CudnnLSTM(int hidden_size,
              int num_layers,
              int batch_size,
              int input_size,
              int num_steps,
              int &params_size) :
                cudnn_handle_(),
				input_desc_({num_steps, batch_size, input_size},
                            {batch_size * input_size, input_size, 1}, num_steps),
				output_desc_({num_steps, batch_size, input_size},
                             {batch_size * input_size, input_size, 1}, num_steps),
				h_init_desc_({num_layers, batch_size, hidden_size},
                             {batch_size * hidden_size, hidden_size, 1}),
				c_init_desc_({num_layers, batch_size, hidden_size},
                             {batch_size * hidden_size, hidden_size, 1}),
				h_final_desc_({num_layers, batch_size, hidden_size},
                              {batch_size * hidden_size, hidden_size, 1}),
				c_final_desc_({num_layers, batch_size, hidden_size},
                              {batch_size * hidden_size, hidden_size, 1}),
                fwd_workspace_({}) {

		dropout_desc_ = new cudnnDropoutDescriptor_t;
		cudnn_status_chk(cudnnCreateDropoutDescriptor(dropout_desc_));
		cudnn_status_chk(cudnnSetDropoutDescriptor(*dropout_desc_,cudnn_handle_.handle(),1,NULL, 0,0));

		rnn_desc_ = RNNDescriptor<float>(hidden_size,
                                         num_layers,
                                         *dropout_desc_,
                                         CUDNN_LINEAR_INPUT,
                                         CUDNN_UNIDIRECTIONAL,
                                         "lstm");
		size_t paramssize;
		cudnn_status_chk(cudnnGetRNNParamsSize(cudnn_handle_.handle(),
											   rnn_desc_.desc(),
                                               input_desc_.desc_elem(),
                                               &paramssize,
                                               CUDNN_DATA_FLOAT));

        weight_desc_ = FilterDescriptorNd<float>(CUDNN_TENSOR_NCHW,
                                                 {int(paramssize/sizeof(float)), 1, 1});

		params_size = int(paramssize/sizeof(float));


        // Set fwd workspace size
        cudnn_status_chk(cudnnGetRNNWorkspaceSize(cudnn_handle_.handle(),
                                                  rnn_desc_.desc(),
                                                  num_steps,
                                                  input_desc_.desc(),
                                                  &fwd_workspace_size_));

        fwd_workspace_ =
            NDArray<float>({static_cast<int>(fwd_workspace_size_ / sizeof(float))});
        fwd_workspace_.device_allocate();
    }


    void forward(NDArray<float>& input,
                 NDArray<float>& h_init,
                 NDArray<float>& c_init,
                 NDArray<float>& h_final,
                 NDArray<float>& c_final,
                 NDArray<float>& weights,
                 NDArray<float>& output,
                 int num_steps) {

        cudnn_status_chk(cudnnRNNForwardInference(cudnn_handle_.handle(),
												 rnn_desc_.desc(),
                                               	 num_steps,
                                                 input_desc_.desc(),
                                                 input.dev_alloc.get(),
												 h_init_desc_.desc(),
                                                 h_init.dev_alloc.get(),
												 c_init_desc_.desc(),
                                                 c_init.dev_alloc.get(),
                                                 weight_desc_.desc(),
                                                 weights.dev_alloc.get(),
                                                 output_desc_.desc(),
                                                 output.dev_alloc.get(),
												 h_final_desc_.desc(),
                                                 h_final.dev_alloc.get(),
												 c_final_desc_.desc(),
                                                 c_final.dev_alloc.get(),
                                                 fwd_workspace_.dev_alloc.get(),
                                                 fwd_workspace_size_));
    }
};


