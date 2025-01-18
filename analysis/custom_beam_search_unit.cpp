#include "custom_beam_search.h"

int main(int argc, char* argv[]) {
    auto BP = parlayANN::BuildParams(
        "CustomBeamSearch",
        64, // max degree
        256, // beam size
        1.2, // alpha
        1, // number of passes
        0, // number of clusters
        0, // cluster size
        0, // MST degree
        0, // delta
        false, // verbose
        false, // quantize build
        0.0, // radius
        0.0, // radius 2
        false, // self
        false, // range
        0 // single batch
    );
 
    return 0;
}
