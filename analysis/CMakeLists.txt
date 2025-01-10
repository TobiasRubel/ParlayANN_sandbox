set(TEST_FILES
    custom_beam_search_unit.cpp
	centrality.cpp
)

foreach(TEST_FILE ${TEST_FILES})
	get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
	add_executable(${TEST_NAME} ${TEST_FILE})
	target_link_libraries(${TEST_NAME} parlay ParlayANN)
endforeach()

#include_directories(${CMAKE_SOURCE_DIR}/parlaylib/include)
#include_directories(${CMAKE_SOURCE_DIR}/algorithms)