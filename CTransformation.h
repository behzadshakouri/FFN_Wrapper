/**
 * @class CTransformation
 * @brief Handles feature-wise normalization and inverse transformation using min-max scaling.
 *
 * Each row represents a feature and each column a sample.
 * Normalization: (x - min) / (max - min)
 */

#ifndef CTRANSFORMATION_H
#define CTRANSFORMATION_H

#include <armadillo>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <iomanip> // for formatting

class CTransformation {
private:
    arma::colvec minValues;  // Minimum values per feature
    arma::colvec maxValues;  // Maximum values per feature

public:
    // ───────────────────────────────────────────────
    // Normalize each row to [0, 1]
    // ───────────────────────────────────────────────
    arma::mat normalize(const arma::mat& data)
    {
        if (data.has_nan())
            std::cerr << "⚠️ [Normalize] Data contains NaN values!" << std::endl;

        if (data.has_inf())
            std::cerr << "⚠️ [Normalize] Data contains Inf values!" << std::endl;

        std::cout << "\n[Normalize] Starting normalization..." << std::endl;
        std::cout << "  Input size: " << data.n_rows << " × " << data.n_cols << std::endl;

        if (data.is_empty())
        {
            std::cerr << "❌ [Normalize] Input data is empty!" << std::endl;
            return data;
        }

        minValues = arma::min(data, 1);
        maxValues = arma::max(data, 1);

        arma::mat normalizedData = data;

        for (arma::uword i = 0; i < data.n_rows; ++i)
        {
            double minVal = minValues(i);
            double maxVal = maxValues(i);
            double range = maxVal - minVal;

            if (!arma::is_finite(minVal) || !arma::is_finite(maxVal) || range <= 1e-12)
            {
                std::cerr << "⚠️ [Normalize] Invalid or zero range at row " << i
                          << " (min=" << minVal << ", max=" << maxVal
                          << "). Setting normalized row to zeros." << std::endl;
                minValues(i) = 0.0;
                maxValues(i) = 1.0;
                normalizedData.row(i).zeros();
                continue;
            }

            normalizedData.row(i) = (data.row(i) - minVal) / range;
        }

        // Debug: print all min and max values
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "[Normalize] Completed." << std::endl;

        std::cout << "  All min values:" << std::endl;
        for (arma::uword i = 0; i < minValues.n_elem; ++i)
            std::cout << "    " << minValues(i);
        std::cout << std::endl;

        std::cout << "  All max values:" << std::endl;
        for (arma::uword i = 0; i < maxValues.n_elem; ++i)
            std::cout << "    " << maxValues(i);
        std::cout << std::endl;

        return normalizedData;
    }

    // ───────────────────────────────────────────────
    // Apply stored min/max normalization
    // ───────────────────────────────────────────────
    arma::mat transform(const arma::mat& data)
    {
        std::cout << "\n[Transform] Applying stored normalization parameters..." << std::endl;

        if (minValues.is_empty() || maxValues.is_empty())
        {
            std::cerr << "⚠️ [Transform] Parameters not loaded — returning unmodified data." << std::endl;
            return data;
        }

        arma::mat normalizedData = data;

        for (arma::uword i = 0; i < data.n_rows; ++i)
        {
            double minVal = minValues(i);
            double maxVal = maxValues(i);
            double range = maxVal - minVal;

            if (!arma::is_finite(minVal) || !arma::is_finite(maxVal) || range <= 1e-12)
            {
                std::cerr << "⚠️ [Transform] Invalid or zero range at row " << i
                          << " — setting row to zeros." << std::endl;
                normalizedData.row(i).zeros();
                continue;
            }

            normalizedData.row(i) = (data.row(i) - minVal) / range;
        }

        std::cout << "[Transform] Done." << std::endl;
        return normalizedData;
    }

    // ───────────────────────────────────────────────
    // Inverse transform: revert normalization
    // ───────────────────────────────────────────────
    arma::mat inverseTransform(const arma::mat& normalizedData)
    {
        std::cout << "\n[InverseTransform] Reverting normalization..." << std::endl;

        if (minValues.is_empty() || maxValues.is_empty())
        {
            throw std::runtime_error("❌ [InverseTransform] Normalization parameters not loaded!");
        }

        arma::mat originalData = normalizedData;
        for (arma::uword i = 0; i < normalizedData.n_rows; ++i)
        {
            double minVal = minValues(i);
            double maxVal = maxValues(i);
            double range = maxVal - minVal;

            originalData.row(i) = normalizedData.row(i) * range + minVal;
        }

        std::cout << "[InverseTransform] Completed successfully." << std::endl;
        return originalData;
    }

    // ───────────────────────────────────────────────
    // Save parameters to text file
    // ───────────────────────────────────────────────
    void saveParameters(const std::string& filename)
    {
        std::cout << "\n[SaveParams] Saving normalization parameters → " << filename << std::endl;
        std::ofstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("❌ [SaveParams] Unable to open file for writing: " + filename);
        }

        for (double val : minValues) file << val << " ";
        file << "\n";
        for (double val : maxValues) file << val << " ";
        file << "\n";

        file.close();
        std::cout << "[SaveParams] Saved " << minValues.n_elem << " min/max pairs." << std::endl;
    }

    // ───────────────────────────────────────────────
    // Load parameters from text file
    // ───────────────────────────────────────────────
    void loadParameters(const std::string& filename)
    {
        std::cout << "\n[LoadParams] Loading normalization parameters ← " << filename << std::endl;
        minValues.clear();
        maxValues.clear();

        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("❌ [LoadParams] Unable to open file for reading: " + filename);
        }

        std::string line;
        std::vector<double> minVals, maxVals;

        if (std::getline(file, line))
        {
            std::stringstream ss(line);
            double val;
            while (ss >> val) minVals.push_back(val);
        }

        if (std::getline(file, line))
        {
            std::stringstream ss(line);
            double val;
            while (ss >> val) maxVals.push_back(val);
        }

        file.close();

        minValues = arma::colvec(minVals);
        maxValues = arma::colvec(maxVals);

        std::cout << "[LoadParams] Loaded " << minValues.n_elem << " parameters." << std::endl;
    }

    // ───────────────────────────────────────────────
    // Getters
    // ───────────────────────────────────────────────
    arma::colvec GetMinValues() const { return minValues; }
    arma::colvec GetMaxValues() const { return maxValues; }
};

#endif // CTRANSFORMATION_H
