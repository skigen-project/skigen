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
