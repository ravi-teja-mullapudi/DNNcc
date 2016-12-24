CXX ?= g++
CXXFLAGS += -g -Wall -std=c++11 -rdynamic

modelio.o: ModelIO.h ModelIO.cpp
	$(CXX) $(CXXFLAGS) ModelIO.cpp -c -o modelio.o

graph.o: Op.h Graph.h Graph.cpp ModelIO.h
	$(CXX) $(CXXFLAGS) Graph.cpp -c -o graph.o

clean:
	rm -rf modelio.o graph.o
