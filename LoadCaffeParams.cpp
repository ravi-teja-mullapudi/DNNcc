#include <caffe/caffe.hpp>
#include "LoadCaffeParams.h"

template<typename T>
NDArray_t convert_blob_to_ndarray(const caffe::Blob<T> &b) {
    switch(b.shape().size()) {
        case 1:
        {
            NDArray<T> arr({b.shape(0)});
            for (int i = 0; i < b.shape(0); i++) {
                arr(i) = b.data_at(i, 0, 0, 0);
            }
            return NDArray_t(arr);
        }
        case 2:
        {
            NDArray<T> arr({b.shape(0), b.shape(1)});
            for (int i = 0; i < b.shape(0); i++) {
                for (int j = 0; j < b.shape(1); j++) {
                    arr(i, j) = b.data_at(i, j, 0, 0);
                }
            }
            return NDArray_t(arr);
        }
        case 3:
        {
            NDArray<T> arr({b.shape(0), b.shape(1), b.shape(2)});
            for (int i = 0; i < b.shape(0); i++) {
                for (int j = 0; j < b.shape(1); j++) {
                    for (int k = 0; k < b.shape(2); k++) {
                        arr(i, j, k) = b.data_at(i, j, k, 0);
                    }
                }
            }
            return NDArray_t(arr);
        }
        case 4:
        {
            NDArray<T> arr({b.shape(0),
                            b.shape(1),
                            b.shape(2),
                            b.shape(3)});

            for (int i = 0; i < b.shape(0); i++) {
                for (int j = 0; j < b.shape(1); j++) {
                    for (int k = 0; k < b.shape(2); k++) {
                        for (int l = 0; l < b.shape(3); l++) {
                            arr(i, j, k, l) = b.data_at(i, j, k, l);
                        }
                    }
                }
            }
            return NDArray_t(arr);
        }
        default:
            std::cout <<
                "Cannot handle a blob with more than 4 dimensions" << std::endl;
            exit(-1);
    }
    return NDArray_t(NDArray<T>());
}

void display_network_info(caffe::Net<float> &net) {
    std::cout << "Num layer names:" << net.layer_names().size() << std::endl;
    for (auto &name: net.layer_names()) {
        std::cout << name << std::endl;
    }
    std::cout << "Num blobs:" << net.blob_names().size() << std::endl;
    for (auto &name: net.blob_names()) {
        std::cout << name << std::endl;
    }
}

void load_params_caffe(char *first_arg,
                        std::string model_file_path,
                        std::string param_file_path,
                        Params& params) {
    // Caffe requires google logging to be initialized
    ::google::InitGoogleLogging(first_arg);
    // Load the network
    caffe::Net<float> net(model_file_path, caffe::TEST);
    net.CopyTrainedLayersFrom(param_file_path);
    // Print network information
    display_network_info(net);
    // Convert caffe blobs into Halide images and populate them
    // into the map of params
    for (size_t i = 0; i < net.layers().size(); i++) {
        for (auto &b: (*net.layers()[i]).blobs()) {
            params[net.layer_names()[i]].push_back(convert_blob_to_ndarray(*b));
        }
    }
}
