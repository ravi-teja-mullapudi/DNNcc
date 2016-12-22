#include <iostream>
#include <assert.h>

struct ConvLayerShape {
    int batch_size;
    int input_width;
    int input_height;
    int input_channels;
    int output_channels;
    int filter_width;
    int filter_height;
    int stride;

    ConvLayerShape(int _batch_size, int _input_width, int _input_height,
                   int _input_channels, int _output_channels,
                   int _filter_width, int _filter_height, int _stride) :
        batch_size(_batch_size),
        input_width(_input_width),
        input_height(_input_height),
        input_channels(_input_channels),
        output_channels(_output_channels),
        filter_width(_filter_width),
        filter_height(_filter_height),
        stride(_stride) {}

    friend std::ostream& operator<<(std::ostream& os, const ConvLayerShape& s) {
        os << "batch_size:" << s.batch_size << ",";
        os << "input_width:" << s.input_width << ",";
        os << "input_height:" << s.input_height << ",";
        os << "input_channels:" << s.input_channels << ",";
        os << "output_channels:" << s.output_channels << ",";
        os << "filter_width:" << s.filter_width << ",";
        os << "filter_height:" << s.filter_height << ",";
        os << "stride:" << s.stride;
        return os;
    }
};
