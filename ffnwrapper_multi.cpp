#include "ffnwrapper_multi.h"
#include <mlpack.hpp>

using namespace mlpack;
using namespace std;
using namespace arma;

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <armadillo>
using namespace mlpack::ann;

#include <QVector>
#include <iostream>
#include <cmath>
#include <gnuplot-iostream.h>

#include <ensmallen.hpp>  // Ensmallen header file

#include <CTransformation.h>


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

    for (int layer = 0; layer<ModelStructure.n_layers; layer++)
    {
        Add<Linear>(ModelStructure.n_nodes[layer]); // Connection Layer : ModelStructure.n_input_layers
        Add<Sigmoid>(); // Activation Funchion
    }

   //model.Add<Linear>(3); // Connection Layer 2: ModelStructure.n_input_layers
    //model.Add<Sigmoid>(); // Activation Funchion 2
    Add<Linear>(TrainOutputData.n_rows); // Output Layer : ModelStructure.n_output_layers
    //Add<Linear>(TrainOutputData.n_rows); // Output Layer : ModelStructure.n_output_layers (Second output)

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

    if (DataCategory == datacategory::Train)
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
        ShiftedInputs.writetofile("ShiftedInputsTrain.txt");
        CTimeSeriesSet<double> ShiftedOutputs = CTimeSeriesSet<double>::OutputShifter(TrainOutputData,ModelStructure.dt,ModelStructure.lags);
        ShiftedOutputs.writetofile("ShiftedOutputsTrain.txt");
    }
    else
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
        ShiftedInputs.writetofile("ShiftedInputsTest.txt");
        CTimeSeriesSet<double> ShiftedOutputs = CTimeSeriesSet<double>::OutputShifter(TestOutputData,ModelStructure.dt,ModelStructure.lags);
        ShiftedOutputs.writetofile("ShiftedOutputsTest.txt");
    }
    return true;
}


bool FFNWrapper_Multi::Transformation()
{

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

    // Train data transform
    CTransformation alldatatransformer;
    arma::mat All_DATA;
    All_DATA = arma::join_rows(TrainInputData, TestInputData);

    //std::cout << "Original Data:\n" << inputdata << std::endl;

    // Normalize train input data
    arma::mat normalizedData = alldatatransformer.normalize(All_DATA);
    //std::cout << "Normalized Data:\n" << normalizedInputData << std::endl;
    normalizedData.save("/home/behzad/Projects/FFNWrapper2/ASM/Results/normalizedidata.txt", arma::file_type::raw_ascii);

    // Save parameters
    alldatatransformer.saveParameters("/home/behzad/Projects/FFNWrapper2/ASM/Results/scaling_params_all.txt");

    // Load parameters for train data
    CTransformation traintransformer;
    traintransformer.loadParameters("/home/behzad/Projects/FFNWrapper2/ASM/Results/scaling_params_all.txt");

    // Load parameters for test
    CTransformation testtransformer;
    testtransformer.loadParameters("/home/behzad/Projects/FFNWrapper2/ASM/Results/scaling_params_all.txt");

    // Test data transform
    arma::mat normalizedTrainData = traintransformer.transform(TrainInputData);
    //std::cout << "Restored Data:\n" << restoredInputData << std::endl;
    normalizedTrainData.save("/home/behzad/Projects/FFNWrapper2/ASM/Results/normalizedtrainidata.txt", arma::file_type::raw_ascii);

    // Test data transform
    arma::mat normalizedTestData = testtransformer.transform(TestInputData);
    //std::cout << "Restored Data:\n" << restoredInputData << std::endl;
    normalizedTestData.save("/home/behzad/Projects/FFNWrapper2/ASM/Results/normalizedtestidata.txt", arma::file_type::raw_ascii);

    // Writing normalized data to matrices
    TrainInputData = normalizedTrainData;
    TestInputData = normalizedTestData;

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


    return true;
}


bool FFNWrapper_Multi::Train()
{

    // Train the model
    FFN::Train(TrainInputData, TrainOutputData);

    // Use the Predict method to get the predictions.
    FFN::Predict(TrainInputData, TrainDataPrediction);
    //cout << "Prediction:" << Prediction;

    return true;
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

    // TrainData
    CTimeSeriesSet<double> TrainDataPrediction1 (TrainDataPrediction,ModelStructure.dt,ModelStructure.lags);
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
    nMSE_Train[constituent] = diff2(TrainDataPrediction1.BTC[constituent],TrainDataTarget.BTC[constituent])/(norm2(TrainDataTarget.BTC[constituent])/TrainDataTarget.BTC[constituent].n);
    _R2_Train[constituent] = R2(TrainDataPrediction1.BTC[constituent],TrainDataTarget.BTC[constituent]);
    }
    // TestData
    CTimeSeriesSet<double> TestDataPrediction1 (TestDataPrediction,ModelStructure.dt,ModelStructure.lags);
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
    nMSE_Test[constituent] = diff2(TestDataPrediction1.BTC[constituent],TestDataTarget.BTC[constituent])/(norm2(TestDataTarget.BTC[constituent])/TestDataTarget.BTC[constituent].n);
    _R2_Test[constituent] = R2(TestDataPrediction1.BTC[constituent],TestDataTarget.BTC[constituent]);
    }
    return true;
}


bool FFNWrapper_Multi::DataSave(datacategory DataCategory) // Saving data
{
    if (silent) return false;

    if (DataCategory==datacategory::Train)
    {   // Input/Output matrix checking
        TrainInputData.save(ModelStructure.outputpath + "TrainInputData.csv", arma::file_type::raw_ascii);
        TrainOutputData.save(ModelStructure.outputpath + "TrainOutputData.csv", arma::file_type::raw_ascii);

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

/*
    // Train and Test Plotter (Output 1)
    for (unsigned int i=0; i<ModelStructure.trainobservedaddress.size(); i++)
    {   CTimeSeriesSet<double> Observed_Train(ModelStructure.trainobservedaddress[i],true);

        CTimeSeriesSet<double> Predicted_Train(ModelStructure.trainpredictedaddress[i],true);

        CTimeSeriesSet<double> Observed_Test(ModelStructure.testobservedaddress[i],true);

        CTimeSeriesSet<double> Predicted_Test(ModelStructure.testpredictedaddress[i],true);

        vector<pair<double, double>> plotdata1, plotdata2, plotdata3, plotdata4;
        for (int i=0; i<Observed_Train.maxnumpoints(); i++)
        {
            plotdata1.push_back(make_pair(Observed_Train.BTC[0].GetT(i),Observed_Train.BTC[0].GetC(i)));

        }
        for (int i=0; i<Predicted_Train.maxnumpoints(); i++)
        {
            plotdata2.push_back(make_pair(Predicted_Train.BTC[0].GetT(i),Predicted_Train.BTC[0].GetC(i)));
        }
        //vector<pair<double, double>> plotdata3, plotdata4;
        for (int i=0; i<Observed_Test.maxnumpoints(); i++)
        {
            plotdata3.push_back(make_pair(Observed_Test.BTC[0].GetT(i),Observed_Test.BTC[0].GetC(i)));

        }
        for (int i=0; i<Predicted_Test.maxnumpoints(); i++)
        {
            plotdata4.push_back(make_pair(Predicted_Test.BTC[0].GetT(i),Predicted_Test.BTC[0].GetC(i)));
        }
        // Create a Gnuplot object
        Gnuplot gp;

        // Set titles and labels
        gp << "set title 'Data Comparison'\n";
        gp << "set xlabel 'Time'\n";
        gp << "set ylabel 'Concentration'\n";
        gp << "set grid\n";  // Optional: Add a grid for better visualization

        // Plot all datasets on the same plot
        gp << "plot '-' with lines title 'Observed Train', '-' with lines title 'Predicted Train'\n, '-' with lines title 'Observed Test'\n, '-' with lines title 'Predicted Test'\n";
        //gp << "plot '.' with lines title 'Observed Test', '.' with lines title 'Predicted Test'\n";
        gp.send1d(plotdata1);  // Send the first dataset (Observed_Train)
        gp.send1d(plotdata2);  // Send the second dataset (Predicted_Train)
        //gp.send1d(plotdata3);  // Send the first dataset (Observed_Test)
        //gp.send1d(plotdata4);  // Send the second dataset (Predicted_Test)
    }

    // Train and Test Plotter (Output 2)
    for (unsigned int i=0; i<ModelStructure.trainobservedaddress.size(); i++)
    {   CTimeSeriesSet<double> Observed_Train(ModelStructure.trainobservedaddress[i],true);

        CTimeSeriesSet<double> Predicted_Train(ModelStructure.trainpredictedaddress[i],true);

        CTimeSeriesSet<double> Observed_Test(ModelStructure.testobservedaddress[i],true);

        CTimeSeriesSet<double> Predicted_Test(ModelStructure.testpredictedaddress[i],true);

        vector<pair<double, double>> plotdata1, plotdata2, plotdata3, plotdata4;
        for (int i=0; i<Observed_Train.maxnumpoints(); i++)
        {
            plotdata1.push_back(make_pair(Observed_Train.BTC[1].GetT(i),Observed_Train.BTC[1].GetC(i)));

        }
        for (int i=0; i<Predicted_Train.maxnumpoints(); i++)
        {
            plotdata2.push_back(make_pair(Predicted_Train.BTC[1].GetT(i),Predicted_Train.BTC[1].GetC(i)));
        }
        //vector<pair<double, double>> plotdata3, plotdata4;
        for (int i=0; i<Observed_Test.maxnumpoints(); i++)
        {
            plotdata3.push_back(make_pair(Observed_Test.BTC[1].GetT(i),Observed_Test.BTC[1].GetC(i)));

        }
        for (int i=0; i<Predicted_Test.maxnumpoints(); i++)
        {
            plotdata4.push_back(make_pair(Predicted_Test.BTC[1].GetT(i),Predicted_Test.BTC[1].GetC(i)));
        }
        // Create a Gnuplot object
        Gnuplot gp;

        // Set titles and labels
        gp << "set title 'Data Comparison'\n";
        gp << "set xlabel 'Time'\n";
        gp << "set ylabel 'Concentration'\n";
        gp << "set grid\n";  // Optional: Add a grid for better visualization

        // Plot all datasets on the same plot
        gp << "plot '-' with lines title 'Observed Train', '-' with lines title 'Predicted Train'\n, '-' with lines title 'Observed Test'\n, '-' with lines title 'Predicted Test'\n";
        //gp << "plot '.' with lines title 'Observed Test', '.' with lines title 'Predicted Test'\n";
        gp.send1d(plotdata1);  // Send the first dataset (Observed_Train)
        gp.send1d(plotdata2);  // Send the second dataset (Predicted_Train)
        //gp.send1d(plotdata3);  // Send the first dataset (Observed_Test)
        //gp.send1d(plotdata4);  // Send the second dataset (Predicted_Test)
    }
*/



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
