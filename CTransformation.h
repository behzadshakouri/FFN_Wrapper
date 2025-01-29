#ifndef CTRANSFORMATION_H
#define CTRANSFORMATION_H

#include <armadillo>
#include <fstream>

class CTransformation {
private:
    arma::rowvec minValues; // Store minimum values of each column
    arma::rowvec maxValues; // Store maximum values of each column

public:
    // Function to normalize each column between 0 and 1
    arma::mat normalize(const arma::mat& data) {
        minValues = arma::min(data, 0); // Compute min for each column
        maxValues = arma::max(data, 0); // Compute max for each column

        arma::mat normalizedData = data;
        for (arma::uword i = 0; i < data.n_cols; ++i) {
            normalizedData.col(i) = (data.col(i) - minValues(i)) / (maxValues(i) - minValues(i));
        }
        return normalizedData;
    }

    arma::mat transform(const arma::mat& data) {
        arma::mat normalizedData = data;
        for (arma::uword i = 0; i < data.n_cols; ++i) {
            normalizedData.col(i) = (data.col(i) - minValues(i)) / (maxValues(i) - minValues(i));
        }
        return normalizedData;
    }

    // Function to perform inverse transformation
    arma::mat inverseTransform(const arma::mat& normalizedData) {
        arma::mat originalData = normalizedData;
        for (arma::uword i = 0; i < normalizedData.n_cols; ++i) {
            originalData.col(i) = normalizedData.col(i) * (maxValues(i) - minValues(i)) + minValues(i);
        }
        return originalData;
    }

    // Function to save scaling parameters to a file using std functions
    void saveParameters(const std::string& filename) {
        std::ofstream file(filename);
        if (file.is_open()) {
            for (double val : minValues) {
                file << val << " ";
            }
            file << std::endl;
            for (double val : maxValues) {
                file << val << " ";
            }
            file << std::endl;
            file.close();
        } else {
            throw std::runtime_error("Unable to open file for writing.");
        }
    }

    // Function to load scaling parameters from a file using std functions
    void loadParameters(const std::string& filename) {
        minValues.clear();
        maxValues.clear();
        std::ifstream file(filename);
        if (file.is_open()) {
            std::string line;
            std::vector<double> minVals, maxVals;

            // Read minValues
            if (std::getline(file, line)) {
                std::stringstream ss(line);
                double val;
                while (ss >> val) {
                    minVals.push_back(val);
                }
            }

            // Read maxValues
            if (std::getline(file, line)) {
                std::stringstream ss(line);
                double val;
                while (ss >> val) {
                    maxVals.push_back(val);
                }
            }

            file.close();

            // Convert vectors to arma::rowvec
            minValues = arma::rowvec(minVals);
            maxValues = arma::rowvec(maxVals);
        } else {
            throw std::runtime_error("Unable to open file for reading.");
        }
    }
    arma::rowvec GetMinValues()
    {
        return minValues;
    }
    arma::rowvec GetMaxValues()
    {
        return maxValues;
    }
};

#endif // CTRANSFORMATION_H
