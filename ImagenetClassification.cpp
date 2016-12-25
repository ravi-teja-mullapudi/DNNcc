#include "networks/Vgg.h"
#include "Graph.h"

int main() {
    Graph g;
    Vgg16(g);
    g.display_ops();
    return 0;
}
