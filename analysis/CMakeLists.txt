set(TEST_FILES
    custom_beam_search_unit.cpp
	centrality.cpp
	beam_shape.cpp
    graph_analysis.cpp
    graph_sparsify.cpp
	deleting_central_edges.cpp
	prune_neighborhood.cpp
	unbounded_all_prune.cpp
	local_shortest_paths.cpp
)

foreach(TEST_FILE ${TEST_FILES})
	get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
	add_executable(${TEST_NAME} ${TEST_FILE})
	target_link_libraries(${TEST_NAME} parlay ParlayANN)
endforeach()

#add_executable(graph_analysis graph_analysis.cpp)
#  target_link_libraries(graph_analysis PRIVATE parlay ParlayANN)
#
#add_executable(graph_sparsify graph_sparsify.cpp)
#  target_link_libraries(graph_sparsify PRIVATE parlay ParlayANN)
