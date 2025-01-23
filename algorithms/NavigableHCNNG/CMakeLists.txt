add_executable(neighbors-nav-hcnng ../bench/neighborsTime.C)
  target_link_libraries(neighbors-nav-hcnng PRIVATE parlay)
  target_precompile_headers(neighbors-nav-hcnng PRIVATE neighbors.h)

