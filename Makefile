CXX ?= g++
CXXFLAGS += -O3 -g -Wall -std=c++11 -rdynamic

CAFFE_PATH = ../caffe/distribute

HALIDE_PATH = /home/ravi/Halide
HALIDE_INC += -I$(HALIDE_PATH)/include -I$(HALIDE_PATH)/tools
HALIDE_LIB += -L$(HALIDE_PATH)/bin -lHalide

CAFFE_INC += -I$(CAFFE_PATH)/include -I/usr/local/cuda/include/
CAFFE_LIB += -L$(CAFFE_PATH)/lib -lcaffe -lglog

BOOST_LIB += -lboost_system -lboost_filesystem

all: classify

modelio.o: ModelIO.h ModelIO.cpp
	$(CXX) $(CXXFLAGS) ModelIO.cpp -c -o modelio.o

op.o: Op.h Op.cpp NDArray.h
	$(CXX) $(CXXFLAGS) Op.cpp -c -o op.o

halide_op.o: Op.h OpImpl.h OpHalide.h OpHalide.cpp
	$(CXX) $(CXXFLAGS) OpHalide.cpp $(HALIDE_INC) -c -o halide_op.o

ref_op.o: Op.h OpImpl.h OpRef.h OpRef.cpp NDArray.h
	$(CXX) $(CXXFLAGS) OpRef.cpp -c -o ref_op.o

graph.o: Op.h OpImpl.h Graph.h Graph.cpp ModelIO.h modelio.o op.o halide_op.o ref_op.o
	$(CXX) $(CXXFLAGS) Graph.cpp -c $(HALIDE_INC) -o graph.o

classify: ImagenetClassification.cpp networks/Vgg.h graph.o op.o halide_op.o ref_op.o
	$(CXX) $(CXXFLAGS) ImagenetClassification.cpp graph.o ref_op.o op.o halide_op.o $(HALIDE_INC) \
					   -I./ $(HALIDE_LIB) -o classify

load_caffe_params.o: LoadCaffeParams.cpp LoadCaffeParams.h ModelIO.h
	$(CXX) $(CXXFLAGS) LoadCaffeParams.cpp -c $(CAFFE_INC) $(CAFFE_LIB) -o load_caffe_params.o

caffe_convert: ConvertCaffeModel.cpp load_caffe_params.o modelio.o
	$(CXX) $(CXXFLAGS) ConvertCaffeModel.cpp load_caffe_params.o modelio.o $(CAFFE_LIB) $(BOOST_LIB)\
					   -o caffe_convert

test_ref: tests/RefGraphTest.cpp graph.o op.o halide_op.o ref_op.o Utils.h
	$(CXX) $(CXXFLAGS) tests/RefGraphTest.cpp graph.o ref_op.o op.o halide_op.o $(HALIDE_INC) \
					   -I./ $(HALIDE_LIB) -o test_ref

test_halide: tests/HalideGraphTest.cpp graph.o op.o halide_op.o ref_op.o Utils.h
	$(CXX) $(CXXFLAGS) tests/HalideGraphTest.cpp graph.o ref_op.o op.o halide_op.o $(HALIDE_INC) \
					   -I./ $(HALIDE_LIB) -o test_halide

test_params: tests/ParamTest.cpp graph.o op.o halide_op.o modelio.o ref_op.o Utils.h networks/Vgg.h
	$(CXX) $(CXXFLAGS) tests/ParamTest.cpp graph.o ref_op.o op.o modelio.o halide_op.o $(HALIDE_INC) \
					   -I./ $(HALIDE_LIB) $(BOOST_LIB) -o test_params

clean:
	rm -rf modelio.o graph.o op.o halide_op.o ref_op.o load_caffe_params.o \
		   classify caffe_convert test_ref test_halide test_params
