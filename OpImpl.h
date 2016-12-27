#pragma once
// TODO: consolidate into a class
// Enumeration of possible implementations of each op node.
enum OpImpl { REF, HALIDE, CUDNN };

// Enumeration of target architecture for each op implementation.
// Currently supports coarse level granularity of CPU/GPU specific
// CPU and GPU arch support to be added in later.
enum TargetArch { CPU, GPU };

// There is a whole hierarchy of programming model targets
// and device capabilities.
enum TargetModel { CUDA, OPENCL, LLVM, C };
