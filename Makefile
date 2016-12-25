CXX ?= g++
CXXFLAGS += -g -Wall -std=c++11 -rdynamic

all: classify

modelio.o: ModelIO.h ModelIO.cpp
	$(CXX) $(CXXFLAGS) ModelIO.cpp -c -o modelio.o

op.o: Op.h Op.cpp NDArray.h
	$(CXX) $(CXXFLAGS) Op.cpp -c -o op.o

graph.o: Op.h Graph.h Graph.cpp ModelIO.h modelio.o op.o
	$(CXX) $(CXXFLAGS) Graph.cpp -c -o graph.o

classify: ImagenetClassification.cpp graph.o op.o networks/Vgg.h
	$(CXX) $(CXXFLAGS) ImagenetClassification.cpp graph.o op.o -I./ -o classify

clean:
	rm -rf modelio.o graph.o op.o classify
