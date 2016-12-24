#include <iostream>
#include <fstream>
#include "ModelIO.h"

void save_model_to_disk(std::string weight_file_name, Params &params) {
    std::ofstream ofs;
    ofs.open(weight_file_name, std::ofstream::out | std::ofstream::trunc |
                               std::ofstream::binary);

    size_t num_params = params.size();
    ofs.write(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    for (auto &w: params) {
        size_t name_len = w.first.size();
        ofs.write(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        ofs.write(const_cast<char*>(w.first.c_str()), name_len);

        size_t num_params = w.second.size();
        ofs.write(reinterpret_cast<char*>(&num_params), sizeof(num_params));

        for (size_t i = 0; i < w.second.size(); i++) {
            NDArray<float> param = w.second[i];
            int dims = param.dimensions();
            ofs.write(reinterpret_cast<char*>(&dims), sizeof(dims));
            switch (param.dimensions()) {
                case 1: {
                    int d0 = param.extent(0);
                    ofs.write(reinterpret_cast<char*>(&d0), sizeof(d0));
                    for (int i = 0; i < param.extent(0); i++) {
                        ofs.write(reinterpret_cast<char*>(&param(i)), sizeof(float));
                    }
                    break;
                }
                case 2: {
                    int d0 = param.extent(0);
                    ofs.write(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1 = param.extent(1);
                    ofs.write(reinterpret_cast<char*>(&d1), sizeof(d1));
                    for (int i = 0; i < param.extent(0); i++) {
                        for (int j = 0; j < param.extent(1); j++) {
                            ofs.write(reinterpret_cast<char*>(&param(i, j)), sizeof(float));
                        }
                    }
                    break;
                }
                case 3: {
                    int d0 = param.extent(0);
                    ofs.write(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1 = param.extent(1);
                    ofs.write(reinterpret_cast<char*>(&d1), sizeof(d1));
                    int d2 = param.extent(2);
                    ofs.write(reinterpret_cast<char*>(&d2), sizeof(d2));
                    for (int i = 0; i < param.extent(0); i++) {
                        for (int j = 0; j < param.extent(1); j++) {
                            for (int k = 0; k < param.extent(2); k++) {
                                ofs.write(reinterpret_cast<char*>(&param(i, j, k)), sizeof(float));
                            }
                        }
                    }
                    break;
                }
                case 4: {
                    int d0 = param.extent(0);
                    ofs.write(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1 = param.extent(1);
                    ofs.write(reinterpret_cast<char*>(&d1), sizeof(d1));
                    int d2 = param.extent(2);
                    ofs.write(reinterpret_cast<char*>(&d2), sizeof(d2));
                    int d3 = param.extent(3);
                    ofs.write(reinterpret_cast<char*>(&d3), sizeof(d3));
                    for (int i = 0; i < param.extent(0); i++) {
                        for (int j = 0; j < param.extent(1); j++) {
                            for (int k = 0; k < param.extent(2); k++) {
                                for (int l = 0; l < param.extent(3); l++) {
                                    ofs.write(reinterpret_cast<char*>(&param(i, j, k, l)), sizeof(float));
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
}

void load_model_from_disk(std::string weight_file_name, Params &params) {
    std::ifstream ifs;
    ifs.open(weight_file_name, std::ifstream::in | std::ofstream::binary);

    size_t num_params;
    ifs.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    for (size_t w = 0; w < num_params; w++) {
        size_t name_len;
        ifs.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

        std::string layer_name;
        layer_name.resize(name_len);
        ifs.read(const_cast<char*>(&layer_name[0]), name_len);

        size_t num_params;
        ifs.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

        for (size_t i = 0; i < num_params; i++) {
            int dims;
            ifs.read(reinterpret_cast<char*>(&dims), sizeof(dims));
            switch (dims) {
                case 1: {
                    int d0;
                    ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
                    NDArray<float> param({d0});
                    for (int i = 0; i < d0; i++) {
                        ifs.read(reinterpret_cast<char*>(&param(i)), sizeof(float));
                    }
                    params[layer_name].push_back(param);
                    break;
                }
                case 2: {
                    int d0;
                    ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1;
                    ifs.read(reinterpret_cast<char*>(&d1), sizeof(d1));
                    NDArray<float> param({d0, d1});
                    for (int i = 0; i < d0; i++) {
                        for (int j = 0; j < d1; j++) {
                            ifs.read(reinterpret_cast<char*>(&param(i, j)), sizeof(float));
                        }
                    }
                    params[layer_name].push_back(param);
                    break;
                }
                case 3: {
                    int d0;
                    ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1;
                    ifs.read(reinterpret_cast<char*>(&d1), sizeof(d1));
                    int d2;
                    ifs.read(reinterpret_cast<char*>(&d2), sizeof(d2));
                    NDArray<float> param({d0, d1, d2});
                    for (int i = 0; i < d0; i++) {
                        for (int j = 0; j < d1; j++) {
                            for (int k = 0; k < d2; k++) {
                                ifs.read(reinterpret_cast<char*>(&param(i, j, k)), sizeof(float));
                            }
                        }
                    }
                    params[layer_name].push_back(param);
                    break;
                }
                case 4: {
                    int d0;
                    ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1;
                    ifs.read(reinterpret_cast<char*>(&d1), sizeof(d1));
                    int d2;
                    ifs.read(reinterpret_cast<char*>(&d2), sizeof(d2));
                    int d3;
                    ifs.read(reinterpret_cast<char*>(&d3), sizeof(d3));
                    NDArray<float> param({d0, d1, d2, d3});
                    for (int i = 0; i < d0; i++) {
                        for (int j = 0; j < d1; j++) {
                            for (int k = 0; k < d2; k++) {
                                for (int l = 0; l < d3; l++) {
                                    ifs.read(reinterpret_cast<char*>(&param(i, j, k, l)), sizeof(float));
                                }
                            }
                        }
                    }
                    params[layer_name].push_back(param);
                    break;
                }
            }
        }
    }
}
