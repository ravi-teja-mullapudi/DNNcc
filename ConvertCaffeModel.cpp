#include "LoadCaffeParams.h"

int main(int argc, char **argv) {
    std::string caffe_proto_file = argv[1];
    std::string caffe_weights_file = argv[2];
    std::string dnncc_weights_file = argv[3];

    Params params;

    load_params_caffe(argv[0],
                      caffe_proto_file,
                      caffe_weights_file,
                      params);

    save_model_to_disk(dnncc_weights_file, params);
    return 0;
}
