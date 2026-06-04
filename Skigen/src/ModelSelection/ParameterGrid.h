// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MODEL_SELECTION_PARAMETER_GRID_H
#define SKIGEN_MODEL_SELECTION_PARAMETER_GRID_H

#include "../Core/Params.h"

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup ModelSelection
/// @{

/// @brief Cartesian-product iterator over a hyperparameter grid.
///
/// Mirrors
/// [sklearn.model_selection.ParameterGrid](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html).
///
/// Each grid maps parameter names to a list of candidate values.
/// `ParameterGrid` expands all combinations and exposes them via
/// `operator[]` and an STL iterator interface.
class ParameterGrid {
public:
    using Grid = std::map<std::string, std::vector<ParameterValue>,
                          std::less<>>;

    explicit ParameterGrid(Grid grid)
        : grid_(std::move(grid)) {
        keys_.reserve(grid_.size());
        for (const auto& [k, v] : grid_) {
            if (v.empty()) {
                throw std::invalid_argument(
                    "ParameterGrid: empty value list for key '" + k + "'");
            }
            keys_.push_back(k);
        }
    }

    [[nodiscard]] std::size_t size() const noexcept {
        if (keys_.empty()) return 0;
        std::size_t n = 1;
        for (const auto& [k, v] : grid_) n *= v.size();
        return n;
    }

    [[nodiscard]] ParameterDict operator[](std::size_t idx) const {
        ParameterDict out;
        std::size_t rem = idx;
        for (auto it = keys_.rbegin(); it != keys_.rend(); ++it) {
            const auto& vals = grid_.at(*it);
            out[*it] = vals[rem % vals.size()];
            rem /= vals.size();
        }
        return out;
    }

    /// @cond INTERNAL
    class Iterator {
    public:
        using value_type = ParameterDict;
        using difference_type = std::ptrdiff_t;
        using reference = ParameterDict;

        Iterator(const ParameterGrid* g, std::size_t i) : g_(g), i_(i) {}
        ParameterDict operator*() const { return (*g_)[i_]; }
        Iterator& operator++() { ++i_; return *this; }
        Iterator operator++(int) { auto t = *this; ++i_; return t; }
        bool operator==(const Iterator& o) const { return i_ == o.i_; }
        bool operator!=(const Iterator& o) const { return i_ != o.i_; }

    private:
        const ParameterGrid* g_;
        std::size_t i_;
    };
    /// @endcond

    [[nodiscard]] Iterator begin() const { return {this, 0}; }
    [[nodiscard]] Iterator end() const { return {this, size()}; }

    [[nodiscard]] const Grid& grid() const noexcept { return grid_; }

private:
    Grid grid_;
    std::vector<std::string> keys_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_MODEL_SELECTION_PARAMETER_GRID_H
