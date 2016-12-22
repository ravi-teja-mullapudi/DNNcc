# DNNcc
TODO
// Inference graph completely implemented in Halide
// What do the right schedules look like and is Halide good enough to express them?
1) Add cuDNN benchmark
2) Add AOT compile for individual kernels
3) Better interface for defining networks, loading weights, and doing back propagation
4) Auto tuning for each layer shape
5) Add a ton of networks for benchmarking
6) Ability to spit out an OP for integration into tensorflow/caffe

// Intermediate representation for ML/deep learning frameworks? What should the frontend
// look like for expressing a wide range of layers?
1) Enable graph to be stitched together from generated kernels and library calls
-- This is useful for comparisons but not certain if Halide extern mechanism
   is the right thing here
2) IR for a simple array language and mapping it to LLVM
-- Kalidescope tutorial on LLVM
-- CUDA kernel generator from IR

// Automatic differentiation and optimization
1) Re-implement opt and proximal in C + Halide + CUDA. Both papers have benchmarks
that are out there.

// Better models for adaptive computation
1) Get training up and running on tensorflow/caffe on popular datasets
2) Clockwork networks
3) Spatially Adaptive Computation Time for Residual Networks

// Better graphing and visualization
1) Plot benchmark results in a super neat way
2) Make benchmark dashboard accessible from the web
