cmake_minimum_required(VERSION 3.20)

add_executable(basic_pipeline basic_pipeline.cpp)
target_link_libraries(basic_pipeline PRIVATE Skigen::Skigen)
target_compile_options(basic_pipeline PRIVATE ${SKIGEN_WARNING_FLAGS})
