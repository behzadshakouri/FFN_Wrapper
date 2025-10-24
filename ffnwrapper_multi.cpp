#include "ffnwrapper_multi.h"

// ────────── mlpack / Armadillo / Ensmallen ──────────
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <ensmallen.hpp>
#include <armadillo>

// ────────── Qt, Standard Library, External ──────────
#include <QVector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <gnuplot-iostream.h>
#include <CTransformation.h>

// ────────── Namespaces ──────────
using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;

FFNWrapper_Multi::FFNWrapper_Multi():FFN<MeanSquaredError>()
{

}

FFNWrapper_Multi::FFNWrapper_Multi(const FFNWrapper_Multi &rhs):FFN<MeanSquaredError>(rhs)
{
    ModelStructure = rhs.ModelStructure;
    TrainInputData = rhs.TrainInputData;
    TrainOutputData = rhs.TrainOutputData;
    TestInputData = rhs.TestInputData;
    TestOutputData = rhs.TestOutputData;

}

FFNWrapper_Multi& FFNWrapper_Multi::operator=(const FFNWrapper_Multi& rhs)
{
    FFN<MeanSquaredError>::operator=(rhs);
    ModelStructure = rhs.ModelStructure;
    TrainInputData = rhs.TrainInputData;
    TrainOutputData = rhs.TrainOutputData;
    TestInputData = rhs.TestInputData;
    TestOutputData = rhs.TestOutputData;

    return *this;
}
FFNWrapper_Multi::~FFNWrapper_Multi()
{

}


bool FFNWrapper_Multi::Initiate(bool dataprocess) // Initiating data
{

    DataProcess();

    //Initialize the network
    if (!dataprocess)
        FFN::operator=(FFN<MeanSquaredError>());

    //ANN
    for (int layer = 0; layer<ModelStructure.n_layers; layer++)
    {
        Add<Linear>(ModelStructure.n_nodes[layer]); // Connection Layer : ModelStructure.n_input_layers
            Add<Sigmoid>(); // Activation Funchion
    }

    Add<ReLU>();
    Add<Linear>(TrainOutputData.n_rows); // Output Layer : ModelStructure.n_output_layers

    return true;
}

bool FFNWrapper_Multi::DataProcess()
{

    Shifter(datacategory::Train);
    Shifter(datacategory::Test);
    Transformation();

    return true;
}



bool FFNWrapper_Multi::Shifter(datacategory DataCategory) // Shifting the data according to lags
{
    segment_sizes.clear();

    if (DataCategory == datacategory::Train) // Train
    {
        TrainInputData.clear();
        TrainOutputData.clear();
        for (unsigned int i=0; i<ModelStructure.trainaddress.size(); i++)
        {   CTimeSeriesSet<double> InputTimeSeries(ModelStructure.trainaddress[i],true);

            //Shifting by lags definition (Inputs)

            mat TrainInputData1 = InputTimeSeries.ToArmaMatShifter(ModelStructure.inputcolumns, ModelStructure.lags);

            //CTimeSeriesSet<double> ShiftedInputs(TrainInputData,ModelStructure.dt,ModelStructure.lags);
            //ShiftedInputs.writetofile("ShiftedInputs.txt");

            //Shifting by lags definition (Outputs)
            mat TrainOutputData1 = InputTimeSeries.ToArmaMatShifterOutput(ModelStructure.outputcolumns, ModelStructure.lags);
            if (ModelStructure.log_output)
                TrainOutputData1 = InputTimeSeries.Log().ToArmaMatShifterOutput(ModelStructure.outputcolumns, ModelStructure.lags); // Log

            if (i==0)
            {
                TrainInputData = TrainInputData1;
                TrainOutputData = TrainOutputData1;
            }
            else
            {
                TrainInputData = arma::join_rows(TrainInputData, TrainInputData1); // Behzad, not sure if it should be join_cols or join_rows, we need to test
                TrainOutputData = arma::join_rows(TrainOutputData, TrainOutputData1);
            }
            segment_sizes.push_back(TrainInputData1.n_cols);

        }

        CTimeSeriesSet<double> ShiftedInputs(TrainInputData,ModelStructure.dt,ModelStructure.lags); // Behzad, This part is to test the shifter. We can comment out after the test.
        ShiftedInputs.writetofile(ModelStructure.outputpath + "ShiftedInputsTrain.txt");
        CTimeSeriesSet<double> ShiftedOutputs = CTimeSeriesSet<double>::OutputShifter(TrainOutputData,ModelStructure.dt,ModelStructure.lags);
        ShiftedOutputs.writetofile(ModelStructure.outputpath + "ShiftedOutputsTrain.txt");
    }
    else //Test
    {
        TestInputData.clear();
        TestOutputData.clear();
        for (unsigned int i=0; i<ModelStructure.testaddress.size(); i++)
        {
            CTimeSeriesSet<double> InputTimeSeries(ModelStructure.testaddress[i],true);

            //Shifting by lags definition (Inputs)

            mat TestInputData1 = InputTimeSeries.ToArmaMatShifter(ModelStructure.inputcolumns, ModelStructure.lags);

            //CTimeSeriesSet<double> ShiftedInputs(TrainInputData,ModelStructure.dt,ModelStructure.lags);
            //ShiftedInputs.writetofile("ShiftedInputs.txt");

            //Shifting by lags definition (Outputs)
            mat TestOutputData1 = InputTimeSeries.ToArmaMatShifterOutput(ModelStructure.outputcolumns, ModelStructure.lags);
            if (ModelStructure.log_output)
                TestOutputData1 = InputTimeSeries.Log().ToArmaMatShifterOutput(ModelStructure.outputcolumns, ModelStructure.lags); // Log

            if (i==0)
            {
                TestInputData = TestInputData1;
                TestOutputData = TestOutputData1;
            }
            else
            {
                TestInputData = arma::join_rows(TestInputData, TestInputData1); // Behzad, not sure if it should be join_cols or join_rows, we need to test
                TestOutputData = arma::join_rows(TestOutputData, TestOutputData1);
            }
            segment_sizes.push_back(TestInputData1.n_cols);
        }

        CTimeSeriesSet<double> ShiftedInputs(TestInputData,ModelStructure.dt,ModelStructure.lags); // Behzad, This part is to test the shifter. We can comment out after the test.
        ShiftedInputs.writetofile(ModelStructure.outputpath + "ShiftedInputsTest.txt");
        CTimeSeriesSet<double> ShiftedOutputs = CTimeSeriesSet<double>::OutputShifter(TestOutputData,ModelStructure.dt,ModelStructure.lags);
        ShiftedOutputs.writetofile(ModelStructure.outputpath + "ShiftedOutputsTest.txt");
    }
    return true;
}


bool FFNWrapper_Multi::Transformation()
{

    // Train data transform
    CTransformation alldatatransformer;
    arma::mat All_DATA;
    All_DATA = arma::join_rows(TrainInputData, TestInputData);

    //std::cout << "Original Data:\n" << inputdata << std::endl;

    // Normalize train input data
    arma::mat normalizedData = alldatatransformer.normalize(All_DATA);
    //std::cout << "Normalized Data:\n" << normalizedInputData << std::endl;
    normalizedData.save(ModelStructure.outputpath + "normalizedidata.txt", arma::file_type::raw_ascii);

    // Save parameters
    alldatatransformer.saveParameters(ModelStructure.outputpath + "scaling_params_all.txt");

    // Load parameters for train data
    CTransformation traintransformer;
    traintransformer.loadParameters(ModelStructure.outputpath + "scaling_params_all.txt");

    // Load parameters for test
    CTransformation testtransformer;
    testtransformer.loadParameters(ModelStructure.outputpath + "scaling_params_all.txt");

    // Train data transform
    arma::mat normalizedTrainData = traintransformer.transform(TrainInputData);
    //std::cout << "Restored Data:\n" << restoredInputData << std::endl;
    normalizedTrainData.save(ModelStructure.outputpath + "normalizedtrainidata.txt", arma::file_type::raw_ascii);

    // Test data transform
    arma::mat normalizedTestData = testtransformer.transform(TestInputData);
    //std::cout << "Restored Data:\n" << restoredInputData << std::endl;
    normalizedTestData.save(ModelStructure.outputpath + "normalizedtestidata.txt", arma::file_type::raw_ascii);

    // Writing normalized data to matrices
    TrainInputData = normalizedTrainData;
    TestInputData = normalizedTestData;

    // --- Post-transform stats ---
    PrintDataStats(TrainInputData, TrainOutputData, "Train (final normalized)");
    PrintDataStats(TrainInputData, TrainOutputData, "Train (final normalized)");

    return true;
}


bool FFNWrapper_Multi::Train()
{

    // Train the model

    mlpack::math::RandomSeed(ModelStructure.seed_number);
    //SGD<> optimizer(/* stepSize = */ 0.01, /* batchSize = */ 32, /* maxIterations = */ 1000, /* tolerance = */ 1e-5, /* shuffle = */ false);

    qDebug() << "[Training] Input:" << TrainInputData.n_rows << "×" << TrainInputData.n_cols
             << "| Output:" << TrainOutputData.n_rows << "×" << TrainOutputData.n_cols << Qt::endl;

    qDebug() << "[Training] Input range:"
             << "[" << TrainInputData.min() << "," << TrainInputData.max() << "]"
             << "| Output range:"
             << "[" << TrainOutputData.min() << "," << TrainOutputData.max() << "]"
             << Qt::endl;

    PrintDataStats(TrainInputData, TrainOutputData, "Train (final normalized)");

    FFN::Train(TrainInputData, TrainOutputData);

    // Use the Predict method to get the predictions.
    FFN::Predict(TrainInputData, TrainDataPrediction);
    //cout << "Prediction:" << Prediction;

    return true;
}


bool FFNWrapper_Multi::Train(const arma::mat& input, const arma::mat& output)
{
    TrainInputData = input;
    TrainOutputData = output;

    return Train();  // Call your existing no-argument version
}


bool FFNWrapper_Multi::Train_Single(bool shuffle)
{
    std::cout << "\n==============================\n";
    std::cout << "   Running Single FFN Train   \n";
    std::cout << "==============================\n";

    mlpack::math::RandomSeed(ModelStructure.seed_number);

    arma::mat X_full = TrainInputData;
    arma::mat Y_full = TrainOutputData;


    arma::mat X = TrainInputData;
    arma::mat Y = TrainOutputData;

    const size_t nSamples = X.n_cols;
    if (nSamples < 2)
    {
        std::cerr << "Error: not enough samples (" << nSamples << ").\n";
        return false;
    }

    // ------------------------------------------------------------
    // 1️⃣ Shuffle (optional)
    // ------------------------------------------------------------
    if (shuffle)
    {
        arma::uvec idx = arma::randperm(nSamples);
        X = X.cols(idx);
        Y = Y.cols(idx);
        idx.save(ModelStructure.outputpath + "shuffle_indices_single.csv", arma::csv_ascii);
        std::cout << "[Info] Data shuffled and saved to shuffle_indices_single.csv\n";
    }

    std::cout << "[Info] Training on raw dataset: "
              << X.n_rows << "×" << X.n_cols
              << " | Outputs: " << Y.n_rows << "×" << Y.n_cols << std::endl;

    // ------------------------------------------------------------
    // 2️⃣ Train the model (raw data)
    // ------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    this->Train(X, Y);  // ✅ no scaling, no transformation

    auto end = std::chrono::high_resolution_clock::now();
    double timeSec = std::chrono::duration<double>(end - start).count();

    std::cout << "[Training Completed in " << timeSec << " s]\n";

    // ------------------------------------------------------------
    // 3️⃣ Predict on training set
    // ------------------------------------------------------------
    arma::mat pred;
    this->Predict(X, pred);

    // ------------------------------------------------------------
    // 4️⃣ Compute metrics directly on raw data
    // ------------------------------------------------------------
    double mse = arma::mean(arma::mean(arma::square(pred - Y)));
    arma::rowvec meanY = arma::mean(Y, 1);
    double SSres = arma::accu(arma::square(pred - Y));
    double SStot = arma::accu(arma::square(Y.each_col() - meanY));
    double r2 = 1.0 - (SSres / (SStot + 1e-12));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nFinal Metrics (raw data):\n";
    std::cout << "  MSE = " << mse << "\n";
    std::cout << "  R²  = " << r2  << "\n";
    std::cout << "  Time = " << timeSec << " s\n";

    // ------------------------------------------------------------
    // 5️⃣ Save predictions and summary
    // ------------------------------------------------------------
    pred.save(ModelStructure.outputpath + "final_pred_single.csv", arma::csv_ascii);

    std::ofstream file(ModelStructure.outputpath + "single_train_results.csv");
    file << "MSE,R2,Time_sec\n" << mse << "," << r2 << "," << timeSec << "\n";
    file.close();

    std::cout << "[Saved] Predictions: final_pred_single.csv\n";
    std::cout << "[Saved] Summary CSV: single_train_results.csv\n";

    return true;
}


// 0 = random K-fold, 1 = expanding window, 2 = fixed ratio (computed as 1 - 1/k)
bool FFNWrapper_Multi::Train_kfold(int n_folds, int splitMode)
{
    if (n_folds < 2)
    {
        std::cerr << "Error: n_folds must be >= 2.\n";
        return false;
    }

    mlpack::math::RandomSeed(ModelStructure.seed_number);

    const size_t nSamples = TrainInputData.n_cols;
    if (nSamples < static_cast<size_t>(n_folds))
    {
        std::cerr << "Error: not enough samples for " << n_folds << " folds.\n";
        return false;
    }

    arma::mat X_full = TrainInputData;
    arma::mat Y_full = TrainOutputData;

    arma::mat X = TrainInputData;
    arma::mat Y = TrainOutputData;

    arma::uvec indices;

    // Shuffle only for random K-fold (mode 0)
    if (splitMode == 0)
    {
        indices = arma::randperm(nSamples);
        X = X.cols(indices);
        Y = Y.cols(indices);
        indices.save(ModelStructure.outputpath + "shuffle_indices.csv", arma::csv_ascii);
        std::cout << "[Info] Random shuffle applied and saved to shuffle_indices.csv\n";
    }

    const double trainRatio = 1.0 - (1.0 / static_cast<double>(n_folds));

    std::vector<double> foldMSE, foldR2, foldTime;
    std::vector<double> trainfoldMSE, trainfoldR2;
    double totalMSE = 0.0, totalR2 = 0.0;
    double traintotalMSE = 0.0, traintotalR2 = 0.0;

    std::cout << "Starting " << n_folds << "-fold cross-validation (mode " << splitMode
              << ", train ratio ≈ " << trainRatio * 100 << "%)...\n";

    for (int fold = 0; fold < n_folds; ++fold)
    {
        arma::mat trainX, trainY, valX, valY;

        // ─────── Split selection ───────
        if (splitMode == 0)
        {
            auto [trainPair, validPair] = KFoldSplit(X, Y, n_folds, fold);
            trainX = trainPair.first;  trainY = trainPair.second;
            valX   = validPair.first;  valY   = validPair.second;
        }
        else if (splitMode == 1)
        {
            auto [trainPair, validPair] = KFoldSplit_TimeSeries(X, Y, n_folds, fold);
            trainX = trainPair.first;  trainY = trainPair.second;
            valX   = validPair.first;  valY   = validPair.second;
        }
        else if (splitMode == 2)
        {
            auto [trainPair, validPair] = KFoldSplit_FixedRatio(X, Y, n_folds, fold, trainRatio);
            trainX = trainPair.first;  trainY = trainPair.second;
            valX   = validPair.first;  valY   = validPair.second;
        }
        else
        {
            std::cerr << "Invalid split mode.\n";
            return false;
        }

        if (trainX.n_cols < 2 || valX.n_cols < 2)
        {
            std::cout << "Skipping fold " << (fold + 1)
                      << " (too few samples)\n";
            continue;
        }

        std::cout << "\nFold " << (fold + 1) << " / " << n_folds
                  << " | Train samples: " << trainX.n_cols
                  << " | Validation samples: " << valX.n_cols << std::endl;

        // ⚠️ Reinitialize model to avoid cumulative training
        FFN::operator=(FFN<MeanSquaredError>());
        Initiate(false); // rebuild architecture with fresh random weights

        // ─────── Train this fold ───────
        auto start = std::chrono::high_resolution_clock::now();
        this->Train(trainX, trainY);
        auto end = std::chrono::high_resolution_clock::now();
        double timeSec = std::chrono::duration<double>(end - start).count();

        // ─────── Evaluate Training ───────
        arma::mat predTrain;
        this->Predict(trainX, predTrain);
        double mseTrain = arma::mean(arma::mean(arma::square(predTrain - trainY)));
        arma::rowvec meanYTrain = arma::mean(trainY, 1);
        double SSresTrain = arma::accu(arma::square(predTrain - trainY));
        double SStotTrain = arma::accu(arma::square(trainY.each_col() - meanYTrain));
        double r2Train = 1.0 - (SSresTrain / (SStotTrain + 1e-12));

        trainfoldMSE.push_back(mseTrain);
        trainfoldR2.push_back(r2Train);
        traintotalMSE += mseTrain;
        traintotalR2  += r2Train;

        std::cout << "  Training  MSE: " << std::setw(10) << mseTrain
                  << " | R²: " << std::setw(8) << r2Train
                  << " | Time: " << timeSec << " s" << std::endl;

        // ─────── Evaluate Validation ───────
        arma::mat predVal;
        this->Predict(valX, predVal);
        double mseVal = arma::mean(arma::mean(arma::square(predVal - valY)));
        arma::rowvec meanYVal = arma::mean(valY, 1);
        double SSresVal = arma::accu(arma::square(predVal - valY));
        double SStotVal = arma::accu(arma::square(valY.each_col() - meanYVal));
        double r2Val = 1.0 - (SSresVal / (SStotVal + 1e-12));

        foldMSE.push_back(mseVal);
        foldR2.push_back(r2Val);
        foldTime.push_back(timeSec);
        totalMSE += mseVal;
        totalR2  += r2Val;

        std::cout << "  Validation MSE: " << std::setw(10) << mseVal
                  << " | R²: " << std::setw(8) << r2Val
                  << " | Time: " << timeSec << " s" << std::endl;
    }

    // ─────── Aggregate Results ───────
    const double avgTrainMSE = traintotalMSE / trainfoldMSE.size();
    const double avgTrainR2  = traintotalR2  / trainfoldR2.size();
    const double avgValMSE   = totalMSE / foldMSE.size();
    const double avgValR2    = totalR2  / foldR2.size();

    std::cout << "\nAverage training   MSE: " << avgTrainMSE
              << " | R²: " << avgTrainR2 << std::endl;
    std::cout << "Average validation MSE: " << avgValMSE
              << " | R²: " << avgValR2 << std::endl;

    // ─────── Save CSV ───────
    const std::string csvPath = ModelStructure.outputpath + "kfold_results.csv";
    std::ofstream file(csvPath);
    file << "Fold,TrainMSE,TrainR2,ValMSE,ValR2,Time_sec\n";
    for (size_t i = 0; i < foldMSE.size(); ++i)
    {
        file << (i + 1) << ","
             << trainfoldMSE[i] << "," << trainfoldR2[i] << ","
             << foldMSE[i] << "," << foldR2[i] << "," << foldTime[i] << "\n";
    }
    file << "Average," << avgTrainMSE << "," << avgTrainR2
         << "," << avgValMSE << "," << avgValR2 << ",-\n";
    file.close();
    std::cout << "Results saved to: " << csvPath << std::endl;

    // ─────── Final full retrain on entire dataset ───────
    std::cout << "Retraining final model on full dataset...\n";
    FFN::operator=(FFN<MeanSquaredError>()); // fresh start again
    Initiate(false);

    this->Train(X_full, Y_full);

    arma::mat fullPred;
    this->Predict(X_full, fullPred);
    fullPred.save(ModelStructure.outputpath + "final_pred_full.csv", arma::csv_ascii);

    double mse_final = arma::mean(arma::mean(arma::square(fullPred - Y_full)));
    arma::rowvec meanY = arma::mean(Y_full, 1);
    double SSres = arma::accu(arma::square(fullPred - Y_full));
    double SStot = arma::accu(arma::square(Y_full.each_col() - meanY));
    double r2_final = 1.0 - (SSres / (SStot + 1e-12));

    std::cout << "\nFinal full-data MSE: " << mse_final
              << " | R²: " << r2_final << std::endl;

    std::ofstream csvAppend(csvPath, std::ios::app);
    csvAppend << "FullDataset," << mse_final << "," << r2_final << ",,,\n";
    csvAppend.close();

    std::cout << "[Done] All results successfully written to: "
              << ModelStructure.outputpath << std::endl;

    return true;
}


std::pair<std::pair<arma::mat, arma::mat>,
          std::pair<arma::mat, arma::mat>>
KFoldSplit(const arma::mat& data,
           const arma::mat& labels,
           size_t k,
           size_t fold)
{
    if (k == 0 || fold >= k)
        throw std::invalid_argument("KFoldSplit: invalid fold or k.");

    const size_t n = data.n_cols;
    const size_t foldSize = n / k;
    const size_t start = fold * foldSize;
    const size_t end   = (fold == k - 1) ? n : start + foldSize;

    arma::uvec valIdx  = arma::regspace<arma::uvec>(start, end - 1);
    arma::uvec mask    = arma::ones<arma::uvec>(n);
    mask(valIdx).zeros();
    arma::uvec trainIdx = arma::find(mask == 1);

    arma::mat trainData   = data.cols(trainIdx);
    arma::mat trainLabels = labels.cols(trainIdx);
    arma::mat validData   = data.cols(valIdx);
    arma::mat validLabels = labels.cols(valIdx);

    return {{trainData, trainLabels}, {validData, validLabels}};
}


std::pair<std::pair<arma::mat, arma::mat>,
         std::pair<arma::mat, arma::mat>>
KFoldSplit_TimeSeries(const arma::mat& data,
                     const arma::mat& labels,
                     size_t k,
                     size_t fold)
{
   if (k < 2) throw std::invalid_argument("KFoldSplit_TimeSeries: k must be >= 2.");
   if (fold >= k) throw std::invalid_argument("KFoldSplit_TimeSeries: fold out of range.");

   const size_t n = data.n_cols;
   const size_t foldSize = n / k;
   const size_t valStart = fold * foldSize;
   const size_t valEnd   = (fold == k - 1) ? n : (fold + 1) * foldSize;

   size_t trainEnd = (valStart == 0) ? foldSize : valStart;
   if (trainEnd < 2) trainEnd = 2;

   arma::mat trainData   = data.cols(0, trainEnd - 1);
   arma::mat trainLabels = labels.cols(0, trainEnd - 1);
   arma::mat validData   = data.cols(valStart, valEnd - 1);
   arma::mat validLabels = labels.cols(valStart, valEnd - 1);

   return {{trainData, trainLabels}, {validData, validLabels}};
}


std::pair<std::pair<arma::mat, arma::mat>,
        std::pair<arma::mat, arma::mat>>
KFoldSplit_FixedRatio(const arma::mat& data,
                    const arma::mat& labels,
                    size_t k,
                size_t fold,
                double trainRatio)
{
  if (k < 2) throw std::invalid_argument("KFoldSplit_FixedRatio: k must be >= 2.");
  if (fold >= k) throw std::invalid_argument("KFoldSplit_FixedRatio: fold out of range.");
  if (trainRatio <= 0.0 || trainRatio >= 1.0)
      throw std::invalid_argument("KFoldSplit_FixedRatio: invalid trainRatio.");

  const size_t n = data.n_cols;
  const size_t foldSize = n / k;
  const size_t valStart = fold * foldSize;
  const size_t valEnd   = (fold == k - 1) ? n : (fold + 1) * foldSize;
  const size_t trainEnd = static_cast<size_t>(trainRatio * n);

  arma::mat trainData   = data.cols(0, trainEnd - 1);
  arma::mat trainLabels = labels.cols(0, trainEnd - 1);
  arma::mat validData   = data.cols(valStart, valEnd - 1);
  arma::mat validLabels = labels.cols(valStart, valEnd - 1);

  return {{trainData, trainLabels}, {validData, validLabels}};
}


bool FFNWrapper_Multi::Test() // Predicting test data
{

    // Use the Predict method to get the predictions.
    FFN::Predict(TestInputData, TestDataPrediction);
    //cout << "Prediction:" << Prediction;


    return true;
}

bool FFNWrapper_Multi::PerformanceMetrics() // Calculating performance metrics
{
    segment_sizes.clear();

    // TrainData
    CTimeSeriesSet<double> TrainDataPrediction1 (TrainDataPrediction,ModelStructure.dt,ModelStructure.lags);
    segment_sizes.push_back(TrainDataPrediction.n_cols);
    vector<CTimeSeriesSet<double>> TrainDataPredictionSplit = CTimeSeriesSet<double>::GetFromArmaMatandSplit(TrainDataPrediction,ModelStructure.dt,ModelStructure.lags,segment_sizes);
    if (!silent)
        for (unsigned int i=0; i<TrainDataPredictionSplit.size(); i++)
            TrainDataPredictionSplit[i].writetofile(ModelStructure.outputpath + "TrainDataPrediction_" + to_string(i) + ".txt");
    CTimeSeriesSet<double> TrainDataTarget = GetTrainOutputData();

    vector<CTimeSeriesSet<double>> TrainDataTargetSplit = CTimeSeriesSet<double>::GetFromArmaMatandSplit(TrainOutputData,ModelStructure.dt,ModelStructure.lags,segment_sizes);
    if (!silent)
        for (unsigned int i=0; i<TrainDataTargetSplit.size(); i++)
            TrainDataTargetSplit[i].writetofile(ModelStructure.outputpath + "TrainDataTarget_" + to_string(i) + ".txt");

    nMSE_Train.resize(ModelStructure.outputcolumns.size());
    _R2_Train.resize(ModelStructure.outputcolumns.size());
    for (int constituent = 0; constituent<ModelStructure.outputcolumns.size(); constituent++)
    {
        double MSE = diff2(TrainDataPrediction1.BTC[constituent],TrainDataTarget.BTC[constituent]);
        double VAR = TrainDataTarget.BTC[constituent].variance();

        nMSE_Train[constituent] = MSE/VAR;
        _R2_Train[constituent] = R2(TrainDataPrediction1.BTC[constituent],TrainDataTarget.BTC[constituent]);
    }

    segment_sizes.clear();

    // TestData
    CTimeSeriesSet<double> TestDataPrediction1 (TestDataPrediction,ModelStructure.dt,ModelStructure.lags);
    segment_sizes.push_back(TestDataPrediction.n_cols);
    vector<CTimeSeriesSet<double>> TestDataPredictionSplit = CTimeSeriesSet<double>::GetFromArmaMatandSplit(TestDataPrediction,ModelStructure.dt,ModelStructure.lags,segment_sizes);
    if (!silent)
        for (unsigned int i=0; i<TestDataPredictionSplit.size(); i++)
            TestDataPredictionSplit[i].writetofile(ModelStructure.outputpath + "TestDataPrediction_" + to_string(i) + ".txt");
    CTimeSeriesSet<double> TestDataTarget = GetTestOutputData();

    vector<CTimeSeriesSet<double>> TestDataTargetSplit = CTimeSeriesSet<double>::GetFromArmaMatandSplit(TestOutputData,ModelStructure.dt,ModelStructure.lags,segment_sizes);
    if (!silent)
        for (unsigned int i=0; i<TestDataTargetSplit.size(); i++)

            TestDataTargetSplit[i].writetofile(ModelStructure.outputpath + "TestDataTarget_" + to_string(i) + ".txt");
    nMSE_Test.resize(ModelStructure.outputcolumns.size());
    _R2_Test.resize(ModelStructure.outputcolumns.size());
    for (int constituent = 0; constituent<ModelStructure.outputcolumns.size(); constituent++)
    {
        double MSE = diff2(TestDataPrediction1.BTC[constituent],TestDataTarget.BTC[constituent]);
        double VAR = TestDataTarget.BTC[constituent].variance();

        nMSE_Test[constituent] = MSE/VAR;
        _R2_Test[constituent] = R2(TestDataPrediction1.BTC[constituent],TestDataTarget.BTC[constituent]);
    }
    return true;
}


bool FFNWrapper_Multi::DataSave(datacategory DataCategory) // Saving data
{
    segment_sizes.clear();

    if (silent) return false;

    if (DataCategory==datacategory::Train)
    {   // Input/Output matrix checking
        TrainInputData.save(ModelStructure.outputpath + "TrainInputData.csv", arma::file_type::raw_ascii);
        TrainOutputData.save(ModelStructure.outputpath + "TrainOutputData.csv", arma::file_type::raw_ascii);

        segment_sizes.push_back(TrainDataPrediction.n_cols);
        vector<CTimeSeriesSet<double>> TrainInputSplit = CTimeSeriesSet<double>::GetFromArmaMatandSplit(TrainInputData,ModelStructure.dt,ModelStructure.lags,segment_sizes);
        for (unsigned int i=0; i<TrainInputSplit.size(); i++)
            TrainInputSplit[i].writetofile(ModelStructure.outputpath + "TrainInputDataTS_" + to_string(i) + ".csv");

        vector<CTimeSeriesSet<double>> TrainOutputSplit = CTimeSeriesSet<double>::GetFromArmaMatandSplit(TrainOutputData,ModelStructure.dt,ModelStructure.lags,segment_sizes);
        for (unsigned int i=0; i<TrainOutputSplit.size(); i++)
            TrainOutputSplit[i].writetofile(ModelStructure.outputpath + "TrainOutputDataTS_" + to_string(i) + ".csv");

        //Prediction results
        TrainDataPrediction.save(ModelStructure.outputpath + "TrainDataPrediction.csv",arma::file_type::raw_ascii);

        vector<CTimeSeriesSet<double>> PredictionSplit = CTimeSeriesSet<double>::GetFromArmaMatandSplit(TrainDataPrediction,ModelStructure.dt,ModelStructure.lags,segment_sizes);
        for (unsigned int i=0; i<PredictionSplit.size(); i++)
            PredictionSplit[i].writetofile(ModelStructure.outputpath + "TrainDataPredictionTS_" + to_string(i) + ".csv");

        TrainInputData.save(ModelStructure.outputpath + "TrainInputData.txt",arma::file_type::raw_ascii);
        TrainOutputData.save(ModelStructure.outputpath + "TrainOutputData.txt",arma::file_type::raw_ascii);

        // Performance metrics
        for (int constituent = 0; constituent<ModelStructure.outputcolumns.size(); constituent++)
        {
        cout<<"nMSE_Train_" + aquiutils::numbertostring(constituent) + "="<<nMSE_Train[constituent]<<endl;
        cout<<"R2_Train_" + aquiutils::numbertostring(constituent) + "="<<_R2_Train[constituent]<<endl;
        }

    }

    else if (DataCategory==datacategory::Test)
    {   //Prediction results
        TestDataPrediction.save(ModelStructure.outputpath + "TestDataPrediction.csv",arma::file_type::raw_ascii);

        segment_sizes.push_back(TestDataPrediction.n_cols);
        vector<CTimeSeriesSet<double>> PredictionSplit = CTimeSeriesSet<double>::GetFromArmaMatandSplit(TestDataPrediction,ModelStructure.dt,ModelStructure.lags,segment_sizes);
        for (unsigned int i=0; i<PredictionSplit.size(); i++)
            PredictionSplit[i].writetofile(ModelStructure.outputpath + "TestDataPredictionTS_" + to_string(i) + ".csv");

        TestInputData.save(ModelStructure.outputpath + "TestInputData.txt",arma::file_type::raw_ascii);
        TestOutputData.save(ModelStructure.outputpath + "TestOutputData.txt",arma::file_type::raw_ascii);

        CTimeSeriesSet<double> TestInputTS(TestInputData,ModelStructure.dt,ModelStructure.lags);
        TestInputTS.writetofile(ModelStructure.outputpath + "TestInputTS_All.csv");

        CTimeSeriesSet<double> TestOutputTS(TestOutputData,ModelStructure.dt,ModelStructure.lags);
        TestOutputTS.writetofile(ModelStructure.outputpath + "TestOutputTS_All.csv");

        vector<CTimeSeriesSet<double>> TestInputSplit = CTimeSeriesSet<double>::GetFromArmaMatandSplit(TestInputData,ModelStructure.dt,ModelStructure.lags,segment_sizes);
        for (unsigned int i=0; i<TestInputSplit.size(); i++)
            TestInputSplit[i].writetofile(ModelStructure.outputpath + "TestInputDataTS_" + to_string(i) + ".csv");


        vector<CTimeSeriesSet<double>> TestOutputSplit = CTimeSeriesSet<double>::GetFromArmaMatandSplit(TestOutputData,ModelStructure.dt,ModelStructure.lags,segment_sizes);
        for (unsigned int i=0; i<TestOutputSplit.size(); i++)
            TestOutputSplit[i].writetofile(ModelStructure.outputpath + "TestOutputDataTS_" + to_string(i) + ".csv");

        // Performance metrics
        for (int constituent = 0; constituent<ModelStructure.outputcolumns.size(); constituent++)
        {
        cout<<"nMSE_Test_" + aquiutils::numbertostring(constituent) + "="<<nMSE_Test[constituent]<<endl;
        cout<<"R2_Test_" + aquiutils::numbertostring(constituent) + "="<<_R2_Test[constituent]<<endl;
        }
    }

    return true;
}


bool FFNWrapper_Multi:: Plotter() // Plotting the results
{

    // Train and Test Plotter (Output 1)
    for (unsigned int i=0; i<ModelStructure.trainobservedaddress.size(); i++)
    {
        CTimeSeriesSet<double> Observed_Train(ModelStructure.trainobservedaddress[i],true);

        CTimeSeriesSet<double> Predicted_Train(ModelStructure.trainpredictedaddress[i],true);

        CTimeSeriesSet<double> Observed_Test(ModelStructure.testobservedaddress[i],true);

        CTimeSeriesSet<double> Predicted_Test(ModelStructure.testpredictedaddress[i],true);

        CTimeSeriesSet<double> Observed = Observed_Train;
        Observed.merge(Observed_Test,true);
        Observed.writetofile(ModelStructure.outputpath + "Observed_Data.csv",false);

        CTimeSeriesSet<double> Predicted = Predicted_Train;
        Predicted.merge(Predicted_Test,true);
        Predicted.writetofile(ModelStructure.outputpath + "Predicted_Data.csv",false);

        vector<pair<double, double>> plotdata1, plotdata2;
        for (int i=0; i<Observed.maxnumpoints(); i++)
        {
            plotdata1.push_back(make_pair(Observed.BTC[0].GetT(i),Observed.BTC[0].GetC(i)));
        }

        for (int i=0; i<Predicted.maxnumpoints(); i++)
        {
            plotdata2.push_back(make_pair(Predicted.BTC[0].GetT(i),Predicted.BTC[0].GetC(i)));
        }


        // Create a Gnuplot object
        Gnuplot gp;

        // Set titles and labels
        gp << "set title 'Data Comparison'\n";
        gp << "set xlabel 'Time'\n";
        gp << "set ylabel 'Concentration'\n";
        gp << "set grid\n";  // Optional: Add a grid for better visualization

        // Plot all datasets on the same plot
        gp << "plot '-' with lines title 'Observed', '-' with lines title 'Predicted'\n";
        gp.send1d(plotdata1);  // Send the first dataset (Observed_Train)
        gp.send1d(plotdata2);  // Send the second dataset (Predicted_Train)

    }

    return true;
}


bool FFNWrapper_Multi::PrintDataStats(const arma::mat& X, const arma::mat& Y, const std::string& tag)
{
    double xmin = X.min();
    double xmax = X.max();
    double ymin = Y.min();
    double ymax = Y.max();

    std::cout << "[" << tag << "] Input range: [" << xmin << ", " << xmax << "]"
              << " | Output range: [" << ymin << ", " << ymax << "]" << std::endl;

    // Also print basic means to detect distribution drift
    std::cout << "[" << tag << "] Input mean: " << arma::mean(arma::vectorise(X))
              << " | Output mean: " << arma::mean(arma::vectorise(Y)) << std::endl;

            return true;
}


bool FFNWrapper_Multi:: Optimizer()
{
/*
    // Define the objective function to minimize (f(x) = x^2 + 4x + 4).
    class QuadraticFunction
    {
    public:
        // Function value at a given point x.
        double Evaluate(const rowvec& parameters)
        {
            // f(x) = x^2 + 4x + 4
            double x = parameters(0);
            return x * x + 4 * x + 4;
        }

        // Gradient of the objective function.
        void Gradient(const rowvec& parameters, rowvec& gradient)
        {
            // Derivative of f(x) = 2x + 4
            double x = parameters(0);
            gradient(0) = 2 * x + 4;
        }
    };
        // Create an instance of the quadratic function.
        QuadraticFunction f;

        // Initial parameters (let's start at x = 10).
        rowvec initialPoint = {10};

        // Create the optimizer (using Stochastic Gradient Descent in this case).
        ens::SGD optimizer(0.1, 1000, 1e-6);

        // Optimize the function using the gradient descent algorithm.
        optimizer.Optimize(f, initialPoint);

        // Output the result.
        std::cout << "Optimal point: " << initialPoint(0) << std::endl;
        std::cout << "Optimal value: " << f.Evaluate(initialPoint) << std::endl;
*/
        return true;
}




/*
    // Train data transform
    CTransformation traintransformer;

    //std::cout << "Original Data:\n" << inputdata << std::endl;

    // Normalize train input data
    arma::mat normalizedTrainData = traintransformer.normalize(TrainInputData);
    //std::cout << "Normalized Data:\n" << normalizedInputData << std::endl;
    normalizedTrainData.save("/home/behzad/Projects/FFNWrapper2/ASM/Results/normalizedtrainidata.txt", arma::file_type::raw_ascii);

    // Save parameters
    traintransformer.saveParameters("/home/behzad/Projects/FFNWrapper2/ASM/Results/scaling_params.txt");

    // Load parameters
    CTransformation testtransformer;
    testtransformer.loadParameters("/home/behzad/Projects/FFNWrapper2/ASM/Results/scaling_params.txt");

    // Test data transform
    arma::mat normalizedTestData = testtransformer.transform(TestInputData);
    //std::cout << "Restored Data:\n" << restoredInputData << std::endl;
    normalizedTestData.save("/home/behzad/Projects/FFNWrapper2/ASM/Results/normalizedtestidata.txt", arma::file_type::raw_ascii);

    // Writing normalized data to matrices
    TrainInputData = normalizedTrainData;
    TestInputData = normalizedTestData;
    */

/*
    //---------------------------------------------Outputs--------------------------------------------
    CTransformation alldatatransformer_OP;
    arma::mat All_DATA_OP;
    All_DATA_OP = arma::join_rows(TrainOutputData, TestOutputData);

    arma::mat normalizedData_OP = alldatatransformer_OP.normalize(All_DATA_OP);
    normalizedData_OP.save("/home/behzad/Projects/FFNWrapper2/ASM/Results/normalizedodata.txt", arma::file_type::raw_ascii);

    alldatatransformer_OP.saveParameters("/home/behzad/Projects/FFNWrapper2/ASM/Results/scaling_params_all_OP.txt");

    CTransformation traintransformer_OP;
    traintransformer_OP.loadParameters("/home/behzad/Projects/FFNWrapper2/ASM/Results/scaling_params_all_OP.txt");

    CTransformation testtransformer_OP;
    testtransformer_OP.loadParameters("/home/behzad/Projects/FFNWrapper2/ASM/Results/scaling_params_all_OP.txt");

    arma::mat normalizedTrainData_OP = traintransformer_OP.transform(TrainOutputData);
    normalizedTrainData_OP.save("/home/behzad/Projects/FFNWrapper2/ASM/Results/normalizedtrainodata.txt", arma::file_type::raw_ascii);

    arma::mat normalizedTestData_OP = testtransformer_OP.transform(TestOutputData);
    normalizedTestData_OP.save("/home/behzad/Projects/FFNWrapper2/ASM/Results/normalizedtestodata.txt", arma::file_type::raw_ascii);

    TrainOutputData = normalizedTrainData_OP;
    TestOutputData = normalizedTestData_OP;
    */




/*
    mat TrainInputData1;
    mat TestInputData1;

    // Normalize train data using Min-Max scaling
    minMaxScaler_te_i.Fit(TrainInputData);        // Fit the scaler to the train data
    minMaxScaler_te_i.Transform(TrainInputData,TrainInputData1);  // Normalize the train data

    // Normalize test data using Standard Scaling (z-score normalization)
    minMaxScaler_te_i.Fit(TestInputData);  // Normalize the test data
    minMaxScaler_te_i.Transform(TestInputData,TestInputData1);  // Normalize the test data

    // Save normalized data (if needed)
    mlpack::data::Save("/home/behzad/Projects/FFNWrapper2/ASM/Results/normalized_testinputdata.csv", TrainInputData1);
    mlpack::data::Save("/home/behzad/Projects/FFNWrapper2/ASM/Results/normalized_testoutputdata.csv", TestInputData1);

    TrainInputData = TrainInputData1;
    TestInputData = TestInputData1;
    */

/*
// Train and Test Plotter (Output 2)
for (unsigned int i=0; i<ModelStructure.trainobservedaddress.size(); i++)
{
    CTimeSeriesSet<double> Observed_Train(ModelStructure.trainobservedaddress[i],true);

    CTimeSeriesSet<double> Predicted_Train(ModelStructure.trainpredictedaddress[i],true);

    CTimeSeriesSet<double> Observed_Test(ModelStructure.testobservedaddress[i],true);

    CTimeSeriesSet<double> Predicted_Test(ModelStructure.testpredictedaddress[i],true);

    CTimeSeriesSet<double> Observed = Observed_Train;
    Observed.merge(Observed_Test,true);
    Observed.writetofile("/home/behzad/Projects/FFNWrapper2/ASM/Results/Observed_try.csv",false);

    CTimeSeriesSet<double> Predicted = Predicted_Train;
    Predicted.merge(Predicted_Test,true);
    Predicted.writetofile("/home/behzad/Projects/FFNWrapper2/ASM/Results/Predicted_try.csv",false);

    vector<pair<double, double>> plotdata1, plotdata2;
    for (int i=0; i<Observed.maxnumpoints(); i++)
    {
        plotdata1.push_back(make_pair(Observed.BTC[1].GetT(i),Observed.BTC[1].GetC(i)));
    }

    for (int i=0; i<Predicted.maxnumpoints(); i++)
    {
        plotdata2.push_back(make_pair(Predicted.BTC[1].GetT(i),Predicted.BTC[1].GetC(i)));
    }


    // Create a Gnuplot object
    Gnuplot gp;

    // Set titles and labels
    gp << "set title 'Data Comparison'\n";
    gp << "set xlabel 'Time'\n";
    gp << "set ylabel 'Concentration'\n";
    gp << "set grid\n";  // Optional: Add a grid for better visualization

    // Plot all datasets on the same plot
    gp << "plot '-' with lines title 'Observed', '-' with lines title 'Predicted'\n";
    gp.send1d(plotdata1);  // Send the first dataset (Observed_Train)
    gp.send1d(plotdata2);  // Send the second dataset (Predicted_Train)

}
*/

/*
    // Train Data Plotter (Output 1)
    for (unsigned int i=0; i<ModelStructure.trainobservedaddress.size(); i++)
    {   CTimeSeriesSet<double> Observed(ModelStructure.trainobservedaddress[i],true);

        CTimeSeriesSet<double> Predicted(ModelStructure.trainpredictedaddress[i],true);

        vector<pair<double, double>> plotdata1, plotdata2;
        for (int i=0; i<Observed.maxnumpoints(); i++)
        {
            plotdata1.push_back(make_pair(Observed.BTC[0].GetT(i),Observed.BTC[0].GetC(i)));

        }
        for (int i=0; i<Predicted.maxnumpoints(); i++)
        {
            plotdata2.push_back(make_pair(Predicted.BTC[0].GetT(i),Predicted.BTC[0].GetC(i)));
        }
        // Create a Gnuplot object
        Gnuplot gp;

        // Set titles and labels
        gp << "set title 'Train Data Comparison'\n";
        gp << "set xlabel 'Time'\n";
        gp << "set ylabel 'Concentration'\n";
        gp << "set grid\n";  // Optional: Add a grid for better visualization

        // Plot both datasets on the same plot
        gp << "plot '-' with lines title 'Observed', '-' with lines title 'Predicted'\n";
        gp.send1d(plotdata1);  // Send the first dataset (Observed)
        gp.send1d(plotdata2);  // Send the second dataset (Predicted)

    }

    // Train Data Plotter (Output 2)
    for (unsigned int i=0; i<ModelStructure.trainobservedaddress.size(); i++)
    {   CTimeSeriesSet<double> Observed(ModelStructure.trainobservedaddress[i],true);

        CTimeSeriesSet<double> Predicted(ModelStructure.trainpredictedaddress[i],true);

        vector<pair<double, double>> plotdata1, plotdata2;
        for (int i=0; i<Observed.maxnumpoints(); i++)
        {
            plotdata1.push_back(make_pair(Observed.BTC[1].GetT(i),Observed.BTC[1].GetC(i)));

        }
        for (int i=0; i<Predicted.maxnumpoints(); i++)
        {
            plotdata2.push_back(make_pair(Predicted.BTC[1].GetT(i),Predicted.BTC[1].GetC(i)));
        }
        // Create a Gnuplot object
        Gnuplot gp;

        // Set titles and labels
        gp << "set title 'Train Data Comparison'\n";
        gp << "set xlabel 'Time'\n";
        gp << "set ylabel 'Concentration'\n";
        gp << "set grid\n";  // Optional: Add a grid for better visualization

        // Plot both datasets on the same plot
        gp << "plot '-' with lines title 'Observed', '-' with lines title 'Predicted'\n";
        gp.send1d(plotdata1);  // Send the first dataset (Observed)
        gp.send1d(plotdata2);  // Send the second dataset (Predicted)

    }

    // Test Data Plotter (Output 1)
    for (unsigned int i=0; i<ModelStructure.testobservedaddress.size(); i++)
    {   CTimeSeriesSet<double> Observed(ModelStructure.testobservedaddress[i],true);

        CTimeSeriesSet<double> Predicted(ModelStructure.testpredictedaddress[i],true);

        vector<pair<double, double>> plotdata1, plotdata2;
        for (int i=0; i<Observed.maxnumpoints(); i++)
        {
            plotdata1.push_back(make_pair(Observed.BTC[0].GetT(i),Observed.BTC[0].GetC(i)));

        }
        for (int i=0; i<Predicted.maxnumpoints(); i++)
        {
            plotdata2.push_back(make_pair(Predicted.BTC[0].GetT(i),Predicted.BTC[0].GetC(i)));
        }
        // Create a Gnuplot object
        Gnuplot gp;

        // Set titles and labels
        gp << "set title 'Test Data Comparison'\n";
        gp << "set xlabel 'Time'\n";
        gp << "set ylabel 'Concentration'\n";
        gp << "set grid\n";  // Optional: Add a grid for better visualization

        // Plot both datasets on the same plot
        gp << "plot '-' with lines title 'Observed', '-' with lines title 'Predicted'\n";
        gp.send1d(plotdata1);  // Send the first dataset (Observed)
        gp.send1d(plotdata2);  // Send the second dataset (Predicted)
    }

    // Test Data Plotter (Output 2)
    for (unsigned int i=0; i<ModelStructure.testobservedaddress.size(); i++)
    {   CTimeSeriesSet<double> Observed(ModelStructure.testobservedaddress[i],true);

        CTimeSeriesSet<double> Predicted(ModelStructure.testpredictedaddress[i],true);

        vector<pair<double, double>> plotdata1, plotdata2;
        for (int i=0; i<Observed.maxnumpoints(); i++)
        {
            plotdata1.push_back(make_pair(Observed.BTC[1].GetT(i),Observed.BTC[1].GetC(i)));

        }
        for (int i=0; i<Predicted.maxnumpoints(); i++)
        {
            plotdata2.push_back(make_pair(Predicted.BTC[1].GetT(i),Predicted.BTC[1].GetC(i)));
        }
        // Create a Gnuplot object
        Gnuplot gp;

        // Set titles and labels
        gp << "set title 'Test Data Comparison'\n";
        gp << "set xlabel 'Time'\n";
        gp << "set ylabel 'Concentration'\n";
        gp << "set grid\n";  // Optional: Add a grid for better visualization

        // Plot both datasets on the same plot
        gp << "plot '-' with lines title 'Observed', '-' with lines title 'Predicted'\n";
        gp.send1d(plotdata1);  // Send the first dataset (Observed)
        gp.send1d(plotdata2);  // Send the second dataset (Predicted)
    }
*/
