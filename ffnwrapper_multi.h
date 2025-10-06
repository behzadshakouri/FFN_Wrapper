#ifndef FFNWrapper_MULTI_H
#define FFNWrapper_MULTI_H
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <vector>
#include <BTCSet.h>
#include "cmodelstructure_multi.h"
#include <gnuplot-iostream.h>

using namespace mlpack;
using namespace std;
using namespace arma;

enum class datacategory {Train, Test};

class FFNWrapper_Multi : FFN<MeanSquaredError>
{
public:
    FFNWrapper_Multi();
    FFNWrapper_Multi(const FFNWrapper_Multi &F);
    FFNWrapper_Multi& operator=(const FFNWrapper_Multi& rhs);
    virtual ~FFNWrapper_Multi();

    bool Initiate(bool dataprocess = true);
    bool DataProcess();
    bool Shifter(datacategory);
    bool Transformation();
    bool Train();
    bool Train_kfold(int n_folds);
    bool Test();
    bool PerformanceMetrics();
    bool DataSave(datacategory);
    bool Plotter();
    bool Optimizer();
    mat A;
    vector<int> segment_sizes;
    CModelStructure_Multi ModelStructure;
    //CTimeSeriesSet<double> *data = nullptr;
    //CTimeSeriesSet<double> *data2 = nullptr;
    bool silent = true;

    CTimeSeriesSet<double> GetTrainInputData()
    {
        return CTimeSeriesSet<double>(TrainInputData,ModelStructure.dt,ModelStructure.lags);
    }
    CTimeSeriesSet<double> GetTrainOutputData()
    {
        return CTimeSeriesSet<double>(TrainOutputData,ModelStructure.dt,ModelStructure.lags);
    }
    CTimeSeriesSet<double> GetTestInputData()
    {
        return CTimeSeriesSet<double>(TestInputData,ModelStructure.dt,ModelStructure.lags);
    }
    CTimeSeriesSet<double> GetTestOutputData()
    {
        return CTimeSeriesSet<double>(TestOutputData,ModelStructure.dt,ModelStructure.lags);
    }

    // Prediction Data
    mat TrainDataPrediction;
    mat TestDataPrediction;

    vector<double> nMSE_Train;
    vector<double> _R2_Train;
    vector<double> nMSE_Test;
    vector<double> _R2_Test;

    //Normalization
    mlpack::data::MinMaxScaler minMaxScaler_tr_i;
    mlpack::data::MinMaxScaler minMaxScaler_tr_o;
    mlpack::data::MinMaxScaler minMaxScaler_te_i;
    mlpack::data::MinMaxScaler minMaxScaler_te_o;

private:
    mat TrainInputData;
    mat TrainOutputData;
    mat TestInputData;
    mat TestOutputData;


};



/**
 * @brief Split data and labels into training and validation subsets for
 *        k-fold cross-validation.
 *
 * This function divides a dataset into @p k equally sized folds and returns
 * the training and validation subsets for a specific fold index.
 *
 * Each column of @p data corresponds to one sample, following the mlpack
 * convention where samples are stored as columns.
 *
 * @param data   Matrix of input features (each column is a sample).
 * @param labels Matrix of corresponding labels (one column per sample).
 * @param k      Total number of folds (must be > 0).
 * @param fold   Index of the validation fold, in the range [0, k-1].
 *
 * @return A nested std::pair structured as:
 *   { {trainData, trainLabels}, {validData, validLabels} }
 *   where:
 *     - trainData  : matrix of training samples
 *     - trainLabels: matrix of corresponding labels
 *     - validData  : matrix of validation samples
 *     - validLabels: matrix of corresponding labels
 *
 * @throws std::invalid_argument if k == 0 or fold >= k.
 *
 * @note
 *  - The data is split by contiguous column ranges (no shuffling). If your
 *    dataset is ordered (e.g., sorted by class), you should randomize it
 *    before calling this function using arma::randperm().
 *  - The last fold may contain slightly more samples if n_cols is not
 *    perfectly divisible by k.
 *
 * @example
 *  arma::mat data, labels;
 *  data.load("X.csv");
 *  labels.load("Y.csv");
 *
 *  auto [trainPair, validPair] = KFoldSplit(data, labels, 5, 0);
 *  arma::mat trainX = trainPair.first;
 *  arma::mat trainY = trainPair.second;
 *  arma::mat validX = validPair.first;
 *  arma::mat validY = validPair.second;
 *
 *  std::cout << "Training samples: " << trainX.n_cols
 *            << ", Validation samples: " << validX.n_cols << std::endl;
 */

std::pair<std::pair<arma::mat, arma::mat>, std::pair<arma::mat, arma::mat>> KFoldSplit(const arma::mat& data,
           const arma::mat& labels,
           size_t k,
           size_t fold);

#endif // FFNWrapper_MULTI_H
