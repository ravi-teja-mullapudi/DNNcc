void init_halide_buf(buffer_t& buf, Tensor* t) {

    assert(t->shape().dims() <= 4);

    int curr_stride = 1;
    for (size_t d = 0; d < t->shape().dims(); d++) {
        int tf_dim = t->shape().dims() - d - 1;
        buf.stride[d] = curr_stride;
        buf.min[d] = 0;
        buf.extent[d] = t->shape().dim_size(tf_dim);
        curr_stride = curr_stride * t->shape().dim_size(tf_dim);
    }

    assert(t->dtype() == DT_FLOAT);

    switch(t->shape().dims()) {
        case 1:
            buf.host = (uint8_t*)t->tensor<float, 1>().data();
            break;
        case 2:
            buf.host = (uint8_t*)t->tensor<float, 2>().data();
            break;
        case 3:
            buf.host = (uint8_t*)t->tensor<float, 3>().data();
            break;
        case 4:
            buf.host = (uint8_t*)t->tensor<float, 4>().data();
            break;
        default:
            assert(0);
    }

    buf.elem_size = 4;
}

void init_halide_buf(buffer_t& buf, const Tensor& t) {
    assert(t.shape().dims() <= 4);
    int curr_stride = 1;
    for (size_t d = 0; d < t.shape().dims(); d++) {
        int tf_dim = t.shape().dims() - d - 1;
        buf.stride[d] = curr_stride;
        buf.min[d] = 0;
        buf.extent[d] = t.shape().dim_size(tf_dim);
        curr_stride = curr_stride * t.shape().dim_size(tf_dim);
    }

    assert(t.dtype() == DT_FLOAT);

    switch(t.shape().dims()) {
        case 1:
            buf.host = (uint8_t*)t.tensor<float, 1>().data();
            break;
        case 2:
            buf.host = (uint8_t*)t.tensor<float, 2>().data();
            break;
        case 3:
            buf.host = (uint8_t*)t.tensor<float, 3>().data();
            break;
        case 4:
            buf.host = (uint8_t*)t.tensor<float, 4>().data();
            break;
        default:
            assert(0);
    }

    buf.elem_size = 4;
}
