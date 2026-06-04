cmake_minimum_required(VERSION 3.20)

add_executable(bench_standard_scaler bench_standard_scaler.cpp)
target_link_libraries(bench_standard_scaler PRIVATE Skigen::Skigen)
target_compile_options(bench_standard_scaler PRIVATE ${SKIGEN_WARNING_FLAGS})

add_executable(bench_linear_model bench_linear_model.cpp)
target_link_libraries(bench_linear_model PRIVATE Skigen::Skigen)
target_compile_options(bench_linear_model PRIVATE ${SKIGEN_WARNING_FLAGS})

add_executable(bench_preprocessing bench_preprocessing.cpp)
target_link_libraries(bench_preprocessing PRIVATE Skigen::Skigen)
target_compile_options(bench_preprocessing PRIVATE ${SKIGEN_WARNING_FLAGS})

add_executable(bench_float_vs_double bench_float_vs_double.cpp)
target_link_libraries(bench_float_vs_double PRIVATE Skigen::Skigen)
target_compile_options(bench_float_vs_double PRIVATE ${SKIGEN_WARNING_FLAGS})

add_executable(bench_ensembles bench_ensembles.cpp)
target_link_libraries(bench_ensembles PRIVATE Skigen::Skigen)
target_compile_options(bench_ensembles PRIVATE ${SKIGEN_WARNING_FLAGS})

add_executable(bench_mlp bench_mlp.cpp)
target_link_libraries(bench_mlp PRIVATE Skigen::Skigen)
target_compile_options(bench_mlp PRIVATE ${SKIGEN_WARNING_FLAGS})
