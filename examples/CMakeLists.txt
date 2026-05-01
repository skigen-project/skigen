cmake_minimum_required(VERSION 3.20)

add_executable(basic_pipeline basic_pipeline.cpp)
target_link_libraries(basic_pipeline PRIVATE Skigen::Skigen)
target_compile_options(basic_pipeline PRIVATE ${SKIGEN_WARNING_FLAGS})

add_executable(ml_pipeline ml_pipeline.cpp)
target_link_libraries(ml_pipeline PRIVATE Skigen::Skigen)
target_compile_options(ml_pipeline PRIVATE ${SKIGEN_WARNING_FLAGS})

add_executable(classification classification.cpp)
target_link_libraries(classification PRIVATE Skigen::Skigen)
target_compile_options(classification PRIVATE ${SKIGEN_WARNING_FLAGS})

add_executable(pca_clustering pca_clustering.cpp)
target_link_libraries(pca_clustering PRIVATE Skigen::Skigen)
target_compile_options(pca_clustering PRIVATE ${SKIGEN_WARNING_FLAGS})
