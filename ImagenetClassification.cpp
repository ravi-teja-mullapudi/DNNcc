#include "networks/Vgg.h"
#include "Graph.h"

int main() {
    Graph g;
    Vgg16(g);
    g.display_ops();
    g.build_forward({"prob"});
    return 0;
}
