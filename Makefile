CXX ?= g++
CXXFLAGS += -g -Wall -std=c++11 -rdynamic

HALIDE_PATH = /home/ravi/Halide
HALIDE_INC += -I$(HALIDE_PATH)/include -I$(HALIDE_PATH)/tools
HALIDE_LIB += -L$(HALIDE_PATH)/bin

all: classify

modelio.o: ModelIO.h ModelIO.cpp
	$(CXX) $(CXXFLAGS) ModelIO.cpp -c -o modelio.o

op.o: Op.h Op.cpp NDArray.h
	$(CXX) $(CXXFLAGS) Op.cpp -c -o op.o

halide_op.o: Op.h OpImpl.h OpHalide.h OpHalide.cpp
	$(CXX) $(CXXFLAGS) OpHalide.cpp $(HALIDE_INC) -c -o halide_op.o

graph.o: Op.h OpImpl.h Graph.h Graph.cpp ModelIO.h modelio.o op.o
	$(CXX) $(CXXFLAGS) Graph.cpp -c -o graph.o

classify: ImagenetClassification.cpp graph.o op.o networks/Vgg.h
	$(CXX) $(CXXFLAGS) ImagenetClassification.cpp graph.o op.o -I./ -o classify

clean:
	rm -rf modelio.o graph.o op.o halide_op.o classify
