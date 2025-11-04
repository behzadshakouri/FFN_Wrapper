#ifndef CTRANSFORMATION_H
#define CTRANSFORMATION_H

#include <armadillo>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>

class CTransformation {
private:
    arma::colvec minValues;  // Minimum values per row (feature)
    arma::colvec maxValues;  // Maximum values per row (feature)

public:
    // Normalize each row to [0, 1]
    arma::mat normalize(const arma::mat& data)
    {
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
            double range = maxValues(i) - minValues(i);
            if (range == 0.0)
            {
                std::cerr << "⚠️ [Normalize] Zero range at row " << i
                          << " (min = " << minValues(i)
                          << ", max = " << maxValues(i)
                          << "). Setting normalized values to 0." << std::endl;
                normalizedData.row(i).zeros();
            }
            else
            {
                normalizedData.row(i) = (data.row(i) - minValues(i)) / range;
            }
        }

        std::cout << "[Normalize] Completed." << std::endl;
        std::cout << "  First 5 min values: " << arma::trans(minValues.head(std::min<arma::uword>(5, minValues.n_elem)));
        std::cout << "  First 5 max values: " << arma::trans(maxValues.head(std::min<arma::uword>(5, maxValues.n_elem)));
        return normalizedData;
    }

    arma::mat transform(const arma::mat& data)
    {
        std::cout << "\n[Transform] Applying stored normalization parameters..." << std::endl;

        if (minValues.is_empty() || maxValues.is_empty())
        {
            throw std::runtime_error("❌ [Transform] Normalization parameters not set. Call normalize() or loadParameters() first.");
        }

        arma::mat normalizedData = data;
        for (arma::uword i = 0; i < data.n_rows; ++i)
        {
            double range = maxValues(i) - minValues(i);
            if (range == 0.0)
            {
                std::cerr << "⚠️ [Transform] Zero range at row " << i
                          << " — setting row to zeros." << std::endl;
                normalizedData.row(i).zeros();
            }
            else
            {
                normalizedData.row(i) = (data.row(i) - minValues(i)) / range;
            }
        }

        std::cout << "[Transform] Done." << std::endl;
        return normalizedData;
    }

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
            double range = maxValues(i) - minValues(i);
            originalData.row(i) = normalizedData.row(i) * range + minValues(i);
        }

        std::cout << "[InverseTransform] Completed successfully." << std::endl;
        return originalData;
    }

    void saveParameters(const std::string& filename)
    {
        std::cout << "\n[SaveParams] Saving normalization parameters → " << filename << std::endl;
        std::ofstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("❌ [SaveParams] Unable to open file for writing: " + filename);
        }

        for (double val : minValues)
            file << val << " ";
        file << "\n";
        for (double val : maxValues)
            file << val << " ";
        file << "\n";

        file.close();
        std::cout << "[SaveParams] Saved " << minValues.n_elem << " min/max pairs." << std::endl;
    }

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
            while (ss >> val)
                minVals.push_back(val);
        }

        if (std::getline(file, line))
        {
            std::stringstream ss(line);
            double val;
            while (ss >> val)
                maxVals.push_back(val);
        }

        file.close();

        minValues = arma::colvec(minVals);
        maxValues = arma::colvec(maxVals);

        std::cout << "[LoadParams] Loaded " << minValues.n_elem << " parameters." << std::endl;
    }

    arma::colvec GetMinValues() const { return minValues; }
    arma::colvec GetMaxValues() const { return maxValues; }
};

#endif // CTRANSFORMATION_H
