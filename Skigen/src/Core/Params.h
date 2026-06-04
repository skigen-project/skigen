// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CORE_PARAMS_H
#define SKIGEN_CORE_PARAMS_H

#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace Skigen {

/// @addtogroup Core
/// @{

/// @brief Heterogeneous value type that backs `set_param` / `get_params`.
///
/// Holds the scalar / string / bool variants that hyperparameter search
/// needs to push and pull through an estimator's reflection API.
using ParameterValue = std::variant<int, double, bool, std::string>;

/// @brief Name → value dictionary returned by `get_params()` and accepted
///   by `set_param(...)` (one key at a time).
using ParameterDict = std::map<std::string, ParameterValue, std::less<>>;

/// @cond INTERNAL
struct UnknownParameter : std::invalid_argument {
    explicit UnknownParameter(std::string_view name)
        : std::invalid_argument(
              std::string("Unknown parameter: ") + std::string(name)) {}
};

struct ParameterTypeMismatch : std::invalid_argument {
    ParameterTypeMismatch(std::string_view name, std::string_view what)
        : std::invalid_argument(
              std::string("Parameter ") + std::string(name) +
              ": " + std::string(what)) {}
};
/// @endcond

namespace internal {

/// @brief Cast a `ParameterValue` to `T`. Performs the same widening
///   conversions a sklearn user would expect (int ↔ double).
template <typename T>
T param_cast(std::string_view name, const ParameterValue& v) {
    if (std::holds_alternative<T>(v)) return std::get<T>(v);

    // Permit int → double / double → int widening.
    if constexpr (std::is_same_v<T, double>) {
        if (std::holds_alternative<int>(v)) {
            return static_cast<double>(std::get<int>(v));
        }
    }
    if constexpr (std::is_same_v<T, int>) {
        if (std::holds_alternative<double>(v)) {
            return static_cast<int>(std::get<double>(v));
        }
    }
    throw ParameterTypeMismatch(name, "wrong value type");
}

}  // namespace internal

/// @}

}  // namespace Skigen

// ---------------------------------------------------------------------------
// SKIGEN_PARAMS(...) macro
//
// Use inside an Estimator class body to register parameters with the
// base-class reflection layer:
//
//     SKIGEN_PARAMS(
//         (alpha,         alpha_,         double),
//         (fit_intercept, fit_intercept_, bool))
//
// Generates `set_param_impl(name, value)` and `get_params_impl()` overrides
// that the CRTP base's `set_param` / `get_params` dispatch into. The macro
// list is parsed by the SKIGEN_PARAMS_FOR_EACH expansion below.
// ---------------------------------------------------------------------------

#define SKIGEN_PARAMS_DETAIL_GET(name, member, type)            \
    out[#name] = ::Skigen::ParameterValue(                      \
        static_cast<type>(this->member));

#define SKIGEN_PARAMS_DETAIL_SET(name, member, type)            \
    if (param_name == #name) {                                  \
        this->member =                                          \
            ::Skigen::internal::param_cast<type>(#name, value); \
        return;                                                 \
    }

// Wrap each entry expansion. The trailing `void()` tames pp-token issues.
#define SKIGEN_PARAMS_DETAIL_EXPAND_GET(entry) \
    SKIGEN_PARAMS_DETAIL_GET entry

#define SKIGEN_PARAMS_DETAIL_EXPAND_SET(entry) \
    SKIGEN_PARAMS_DETAIL_SET entry

// Hand-rolled FOR_EACH (variadic) supporting up to 18 entries.
#define SKIGEN_PARAMS_FE_1(M, x)              M(x)
#define SKIGEN_PARAMS_FE_2(M, x, ...)         M(x) SKIGEN_PARAMS_FE_1(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_3(M, x, ...)         M(x) SKIGEN_PARAMS_FE_2(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_4(M, x, ...)         M(x) SKIGEN_PARAMS_FE_3(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_5(M, x, ...)         M(x) SKIGEN_PARAMS_FE_4(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_6(M, x, ...)         M(x) SKIGEN_PARAMS_FE_5(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_7(M, x, ...)         M(x) SKIGEN_PARAMS_FE_6(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_8(M, x, ...)         M(x) SKIGEN_PARAMS_FE_7(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_9(M, x, ...)         M(x) SKIGEN_PARAMS_FE_8(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_10(M, x, ...)        M(x) SKIGEN_PARAMS_FE_9(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_11(M, x, ...)        M(x) SKIGEN_PARAMS_FE_10(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_12(M, x, ...)        M(x) SKIGEN_PARAMS_FE_11(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_13(M, x, ...)        M(x) SKIGEN_PARAMS_FE_12(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_14(M, x, ...)        M(x) SKIGEN_PARAMS_FE_13(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_15(M, x, ...)        M(x) SKIGEN_PARAMS_FE_14(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_16(M, x, ...)        M(x) SKIGEN_PARAMS_FE_15(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_17(M, x, ...)        M(x) SKIGEN_PARAMS_FE_16(M, __VA_ARGS__)
#define SKIGEN_PARAMS_FE_18(M, x, ...)        M(x) SKIGEN_PARAMS_FE_17(M, __VA_ARGS__)

#define SKIGEN_PARAMS_NARG_HELPER(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, N, ...) N
#define SKIGEN_PARAMS_NARG(...) \
    SKIGEN_PARAMS_NARG_HELPER(__VA_ARGS__, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#define SKIGEN_PARAMS_FE_PICK_HELPER(N) SKIGEN_PARAMS_FE_##N
#define SKIGEN_PARAMS_FE_PICK(N) SKIGEN_PARAMS_FE_PICK_HELPER(N)

#define SKIGEN_PARAMS_FOR_EACH(M, ...) \
    SKIGEN_PARAMS_FE_PICK(SKIGEN_PARAMS_NARG(__VA_ARGS__))(M, __VA_ARGS__)

/// @brief Register a parameter list with the reflection layer.
///
/// Each entry is a 3-token tuple: `(public_name, member_field, c++type)`.
/// Generates `set_param_impl(...)` and `get_params_impl()` member functions
/// — the CRTP base's `set_param(name, value)` and `get_params()` dispatch
/// into them.
#define SKIGEN_PARAMS(...)                                                 \
    void set_param_impl(std::string_view param_name,                       \
                        const ::Skigen::ParameterValue& value) {           \
        SKIGEN_PARAMS_FOR_EACH(SKIGEN_PARAMS_DETAIL_EXPAND_SET, __VA_ARGS__) \
        throw ::Skigen::UnknownParameter(param_name);                       \
    }                                                                      \
    ::Skigen::ParameterDict get_params_impl() const {                      \
        ::Skigen::ParameterDict out;                                       \
        SKIGEN_PARAMS_FOR_EACH(SKIGEN_PARAMS_DETAIL_EXPAND_GET, __VA_ARGS__) \
        return out;                                                        \
    }

#endif  // SKIGEN_CORE_PARAMS_H
