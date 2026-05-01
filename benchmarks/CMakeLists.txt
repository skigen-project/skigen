cmake_minimum_required(VERSION 3.20)

add_executable(bench_standard_scaler bench_standard_scaler.cpp)
target_link_libraries(bench_standard_scaler PRIVATE Skigen::Skigen)
target_compile_options(bench_standard_scaler PRIVATE ${SKIGEN_WARNING_FLAGS})
