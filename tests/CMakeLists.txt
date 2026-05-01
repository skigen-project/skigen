cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Test executable
# ---------------------------------------------------------------------------
add_executable(skigen_tests
    test_standard_scaler.cpp
)

target_link_libraries(skigen_tests PRIVATE Skigen::Skigen)
target_compile_options(skigen_tests PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_tests COMMAND skigen_tests)

# ---------------------------------------------------------------------------
# Preprocessing tests (MinMaxScaler, MaxAbsScaler, RobustScaler, etc.)
# ---------------------------------------------------------------------------
add_executable(skigen_test_preprocessing
    test_preprocessing.cpp
)

target_link_libraries(skigen_test_preprocessing PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_preprocessing PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_preprocessing COMMAND skigen_test_preprocessing)

# ---------------------------------------------------------------------------
# Linear model tests (LinearRegression, Ridge)
# ---------------------------------------------------------------------------
add_executable(skigen_test_linear_model
    test_linear_model.cpp
)

target_link_libraries(skigen_test_linear_model PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_linear_model PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_linear_model COMMAND skigen_test_linear_model)
