add_executable(neighbors-nav-hcnng ../bench/neighborsTime.C)
target_link_libraries(neighbors-nav-hcnng PRIVATE parlay ParlayANN)
target_include_directories(neighbors-nav-hcnng PRIVATE ../bench)
target_precompile_headers(neighbors-nav-hcnng PRIVATE neighbors.h)
