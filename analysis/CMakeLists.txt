set(TEST_FILES
    custom_beam_search_unit.cpp
	centrality.cpp
    graph_analysis.cpp
    graph_sparsify.cpp
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
