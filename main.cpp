//Our main

#include <mlpack.hpp>
#include <iostream>
#include "ffnwrapper_multi.h"
#include <BTCSet.h>
#include "modelcreator.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include "ga.h"
#include <CTransformation.h>

using namespace mlpack;
using namespace std;
using namespace arma;


int main()
{
    // Simulation & Data Configuration ---> Should be defined

    const double Realization = 1; // Number of Realizations
    double total_data_cols; // Number of Inputs + Outputs
    double number_of_outputs; // Number of Outputs

    enum class _model {Settling, ASM} model = _model::ASM;

    if (model==_model::ASM)
    {

    }

    bool ASM = true; // true for ASM and false for Settling element simple model

    if (ASM)
    {
    total_data_cols = 10; // Number of Inputs + Outputs (3+1+4+1)+(1)
    number_of_outputs = 1; // Number of Outputs
    }
    else
    {
    total_data_cols = 4; // Number of Inputs + Outputs (1+2)+1
    number_of_outputs = 2; // Number of Outputs
    }

    string data_name = "NO"; // NO, NH, ND, sCOD, VSS, TKN
    bool log_output_d = false; // true for log output data and false for normal output data

    double Seed_number = 42; // 42 is a random number

    bool GA = false;  // true for Genetic Alghorithm usage and false for no Genetic Alghorithm usage
    const double GA_Nsim = 100; // Number of GA simulations ???
    bool MSE_Test = true; // true for MSE_Test minimization and false for (MSE_Test + MSE_Train) minimization
    bool optimized_structure = true; // true for GA optimized network structures and false for my own structure

    bool randommodelstructure = false; // true for random model structure usage and false for no random model structure usage
    const double Random_Nsim = 1000; // Number of random model structure simulations

    //------------------------------------------------------------------------------------------------------------------------------

    //Model creator (Random model structure)
    ModelCreator modelCreator;
    modelCreator.lag_frequency = 3;
    modelCreator.maximum_superficial_lag = 5;
    modelCreator.total_number_of_columns = total_data_cols-number_of_outputs; // Inputs
    modelCreator.max_number_of_layers = 5; // 4 to 6
    modelCreator.max_lag_multiplier = 10;
    modelCreator.max_number_of_nodes_in_layers = 10*4; // 10 to 15


    string path;
    string path_ASM;

#ifdef Arash
    path = "/home/arash/Projects/FFNWrapper/";
    path_ASM = "/home/arash/Projects/FFNWrapper/ASM/";
    string datapath = "/home/arash/Projects/FFNWrapper/";
    string datapath_ASM = "/home/arash/Projects/FFNWrapper/ASM/";
    string buildpath = "build/Desktop_Qt_5_15_2_GCC_64bit-Debug/";
#else
    path = "/home/behzad/Projects/FFNWrapper2/";
    path_ASM = "/home/behzad/Projects/FFNWrapper2/ASM/";
    string datapath = "/home/behzad/Projects/FFNWrapper2/";
    string datapath_ASM = "/home/behzad/Projects/FFNWrapper2/ASM/";
    string buildpath = "build/Desktop_Qt_5_15_2_GCC_64bit-Debug/";
#endif


    // Defining Model Structure
    CModelStructure_Multi mymodelstruct; //randommodelstructure

    if (ASM)
    {
    // ------------------------simple model properties for optimized structure----------------------------------------------------
    mymodelstruct.dt=0.1;
    mymodelstruct.log_output=log_output_d;
    mymodelstruct.realization=Realization;
    mymodelstruct.seed_number=Seed_number;

    if (!optimized_structure)
    {
    // Network structure
    mymodelstruct.n_layers = 3;
    mymodelstruct.n_nodes = {10*4,8*4,7*4};

    // Defining Inputs
    for (int i=0; i<total_data_cols-number_of_outputs; i++)
    {
        mymodelstruct.inputcolumns.push_back(i); // Input 0: Inflow
    }

    // Defining Output(s)
    for (int i = 0; i<number_of_outputs; i++)
        mymodelstruct.outputcolumns.push_back(total_data_cols-(i+1)); // Output: Settling element (1)_Solids:concentration


    // Lags definition
    vector<int> lag0; lag0.push_back(1);
    vector<int> lag1; lag1.push_back(1);
    vector<int> lag2; lag2.push_back(1);
    vector<int> lag3; lag3.push_back(1);
    vector<int> lag4; lag4.push_back(1);
    vector<int> lag5; lag5.push_back(1);
    vector<int> lag6; lag6.push_back(1);
    vector<int> lag7; lag7.push_back(1);
    vector<int> lag8; lag8.push_back(1);

    mymodelstruct.lags.push_back(lag0);
    mymodelstruct.lags.push_back(lag1);
    mymodelstruct.lags.push_back(lag2);
    mymodelstruct.lags.push_back(lag3);
    mymodelstruct.lags.push_back(lag4);
    mymodelstruct.lags.push_back(lag5);
    mymodelstruct.lags.push_back(lag6);
    mymodelstruct.lags.push_back(lag7);
    mymodelstruct.lags.push_back(lag8);
    }

    //------------------------------------------Manual approach for all constituents GA optimized structures

    if (data_name == "NO") // ----------------------------------
    {
        // Network structure
        mymodelstruct.n_layers = 2;
        mymodelstruct.n_nodes = {39,17}; //{10*4,8*4,7*4}

        // Defining Inputs
        mymodelstruct.inputcolumns.push_back(0); //
        //mymodelstruct.inputcolumns.push_back(1); //
        mymodelstruct.inputcolumns.push_back(2); //
        mymodelstruct.inputcolumns.push_back(3); //
        //mymodelstruct.inputcolumns.push_back(4); //
        //mymodelstruct.inputcolumns.push_back(5); //
        mymodelstruct.inputcolumns.push_back(6); //
        mymodelstruct.inputcolumns.push_back(7); //
        mymodelstruct.inputcolumns.push_back(8); //

        // Defining Output(s)
        for (int i = 0; i<number_of_outputs; i++)
            mymodelstruct.outputcolumns.push_back(total_data_cols-(i+1)); // Output: Settling element (1)_Solids:concentration

        // Lags definition
        vector<int> lag0; lag0.push_back(0); lag0.push_back(22); lag0.push_back(33);
        //vector<int> lag1; lag1.push_back(0);
        vector<int> lag2; lag2.push_back(44);
        vector<int> lag3; lag3.push_back(0); lag3.push_back(33); lag3.push_back(44);
        //vector<int> lag4; lag4.push_back(0);
        //vector<int> lag5; lag5.push_back(0);
        vector<int> lag6; lag6.push_back(0); lag6.push_back(11); lag6.push_back(22);
        vector<int> lag7; lag7.push_back(44);
        vector<int> lag8; lag8.push_back(11); lag8.push_back(22);

        mymodelstruct.lags.push_back(lag0);
        //mymodelstruct.lags.push_back(lag1);
        mymodelstruct.lags.push_back(lag2);
        mymodelstruct.lags.push_back(lag3);
        //mymodelstruct.lags.push_back(lag4);
        //mymodelstruct.lags.push_back(lag5);
        mymodelstruct.lags.push_back(lag6);
        mymodelstruct.lags.push_back(lag7);
        mymodelstruct.lags.push_back(lag8);
    }

    else if (data_name == "NH") // ----------------------------------
    {
        // Network structure
        mymodelstruct.n_layers = 3;
        mymodelstruct.n_nodes = {19,8,4}; //{10*4,8*4,7*4}

        // Defining Inputs
        mymodelstruct.inputcolumns.push_back(0); //
        //mymodelstruct.inputcolumns.push_back(1); //
        //mymodelstruct.inputcolumns.push_back(2); //
        mymodelstruct.inputcolumns.push_back(3); //
        mymodelstruct.inputcolumns.push_back(4); //
        //mymodelstruct.inputcolumns.push_back(5); //
        mymodelstruct.inputcolumns.push_back(6); //
        mymodelstruct.inputcolumns.push_back(7); //
        //mymodelstruct.inputcolumns.push_back(8); //

        // Defining Output(s)
        for (int i = 0; i<number_of_outputs; i++)
            mymodelstruct.outputcolumns.push_back(total_data_cols-(i+1)); // Output: Settling element (1)_Solids:concentration

        // Lags definition
        vector<int> lag0; lag0.push_back(1);
        //vector<int> lag1; lag1.push_back(0);
        //vector<int> lag2; lag2.push_back(0);
        vector<int> lag3; lag3.push_back(1); lag3.push_back(3);
        vector<int> lag4; lag4.push_back(2);
        //vector<int> lag5; lag5.push_back(0);
        vector<int> lag6; lag6.push_back(0); lag6.push_back(2);
        vector<int> lag7; lag7.push_back(1);
        //vector<int> lag8; lag8.push_back(0);

        mymodelstruct.lags.push_back(lag0);
        //mymodelstruct.lags.push_back(lag1);
        //mymodelstruct.lags.push_back(lag2);
        mymodelstruct.lags.push_back(lag3);
        mymodelstruct.lags.push_back(lag4);
        //mymodelstruct.lags.push_back(lag5);
        mymodelstruct.lags.push_back(lag6);
        mymodelstruct.lags.push_back(lag7);
        //mymodelstruct.lags.push_back(lag8);
    }

    else if (data_name == "sCOD") // ----------------------------------
    {
        // Network structure
        mymodelstruct.n_layers = 3;
        mymodelstruct.n_nodes = {36,37,7}; //{10*4,8*4,7*4}

        // Defining Inputs
        mymodelstruct.inputcolumns.push_back(0); //
        //mymodelstruct.inputcolumns.push_back(0); //
        mymodelstruct.inputcolumns.push_back(2); //
        mymodelstruct.inputcolumns.push_back(3); //
        //mymodelstruct.inputcolumns.push_back(0); //
        mymodelstruct.inputcolumns.push_back(5); //
        mymodelstruct.inputcolumns.push_back(6); //
        mymodelstruct.inputcolumns.push_back(7); //
        mymodelstruct.inputcolumns.push_back(8); //

        // Defining Output(s)
        for (int i = 0; i<number_of_outputs; i++)
            mymodelstruct.outputcolumns.push_back(total_data_cols-(i+1)); // Output: Settling element (1)_Solids:concentration

        // Lags definition
        vector<int> lag0; lag0.push_back(0); lag0.push_back(1); lag0.push_back(3);
        //vector<int> lag1; lag1.push_back(0);
        vector<int> lag2; lag2.push_back(0); lag2.push_back(1); lag2.push_back(2); lag2.push_back(4);
        vector<int> lag3; lag3.push_back(0); lag3.push_back(2);
        //vector<int> lag4; lag4.push_back(0);
        vector<int> lag5; lag5.push_back(0); lag5.push_back(1); lag5.push_back(2);
        vector<int> lag6; lag6.push_back(2);
        vector<int> lag7; lag7.push_back(0); lag7.push_back(1); lag7.push_back(3);
        vector<int> lag8; lag8.push_back(0); lag8.push_back(2); lag8.push_back(3);

        mymodelstruct.lags.push_back(lag0);
        //mymodelstruct.lags.push_back(lag1);
        mymodelstruct.lags.push_back(lag2);
        mymodelstruct.lags.push_back(lag3);
        //mymodelstruct.lags.push_back(lag4);
        mymodelstruct.lags.push_back(lag5);
        mymodelstruct.lags.push_back(lag6);
        mymodelstruct.lags.push_back(lag7);
        mymodelstruct.lags.push_back(lag8);
    }

    else if (data_name == "TKN") // ----------------------------------
    {
        // Network structure
        mymodelstruct.n_layers = 3;
        mymodelstruct.n_nodes = {11,8,9}; //{10*4,8*4,7*4}

        // Defining Inputs
        mymodelstruct.inputcolumns.push_back(0); //
        //mymodelstruct.inputcolumns.push_back(1); //
        //mymodelstruct.inputcolumns.push_back(2); //
        mymodelstruct.inputcolumns.push_back(3); //
        //mymodelstruct.inputcolumns.push_back(4); //
        mymodelstruct.inputcolumns.push_back(5); //
        mymodelstruct.inputcolumns.push_back(6); //
        mymodelstruct.inputcolumns.push_back(7); //
        mymodelstruct.inputcolumns.push_back(8); //

        // Defining Output(s)
        for (int i = 0; i<number_of_outputs; i++)
            mymodelstruct.outputcolumns.push_back(total_data_cols-(i+1)); // Output: Settling element (1)_Solids:concentration

        // Lags definition
        vector<int> lag0; lag0.push_back(1);
        //vector<int> lag1; lag1.push_back(0);
        //vector<int> lag2; lag2.push_back(0);
        vector<int> lag3; lag3.push_back(1); lag3.push_back(2); lag3.push_back(3);
        //vector<int> lag4; lag4.push_back(0);
        vector<int> lag5; lag5.push_back(0);
        vector<int> lag6; lag6.push_back(0); lag6.push_back(3); lag6.push_back(4);
        vector<int> lag7; lag7.push_back(4);
        vector<int> lag8; lag8.push_back(0); lag8.push_back(4);

        mymodelstruct.lags.push_back(lag0);
        //mymodelstruct.lags.push_back(lag1);
        //mymodelstruct.lags.push_back(lag2);
        mymodelstruct.lags.push_back(lag3);
        //mymodelstruct.lags.push_back(lag4);
        mymodelstruct.lags.push_back(lag5);
        mymodelstruct.lags.push_back(lag6);
        mymodelstruct.lags.push_back(lag7);
        mymodelstruct.lags.push_back(lag8);
    }

    else if (data_name == "VSS") // ----------------------------------
    {
        // Network structure
        mymodelstruct.n_layers = 3;
        mymodelstruct.n_nodes = {11,5,2}; //{10*4,8*4,7*4}

        // Defining Inputs
        mymodelstruct.inputcolumns.push_back(0); //
        //mymodelstruct.inputcolumns.push_back(1); //
        //mymodelstruct.inputcolumns.push_back(2); //
        //mymodelstruct.inputcolumns.push_back(3); //
        //mymodelstruct.inputcolumns.push_back(4); //
        //mymodelstruct.inputcolumns.push_back(5); //
        //mymodelstruct.inputcolumns.push_back(6); //
        mymodelstruct.inputcolumns.push_back(7); //
        //mymodelstruct.inputcolumns.push_back(8); //

        // Defining Output(s)
        for (int i = 0; i<number_of_outputs; i++)
            mymodelstruct.outputcolumns.push_back(total_data_cols-(i+1)); // Output: Settling element (1)_Solids:concentration

        // Lags definition
        vector<int> lag0; lag0.push_back(0); lag0.push_back(12); lag0.push_back(60);
        //vector<int> lag1; lag1.push_back(0);
        //vector<int> lag2; lag2.push_back(0);
        //vector<int> lag3; lag3.push_back(0);
        //vector<int> lag4; lag4.push_back(0);
        //vector<int> lag5; lag5.push_back(0);
        //vector<int> lag6; lag6.push_back(0);
        vector<int> lag7; lag7.push_back(0); lag7.push_back(48);
        //vector<int> lag8; lag8.push_back(0);

        mymodelstruct.lags.push_back(lag0);
        //mymodelstruct.lags.push_back(lag1);
        //mymodelstruct.lags.push_back(lag2);
        //mymodelstruct.lags.push_back(lag3);
        //mymodelstruct.lags.push_back(lag4);
        //mymodelstruct.lags.push_back(lag5);
        //mymodelstruct.lags.push_back(lag6);
        mymodelstruct.lags.push_back(lag7);
        //mymodelstruct.lags.push_back(lag8);
    }

    else if (data_name == "ND") // ----------------------------------
    {
        // Network structure
        mymodelstruct.n_layers = 3;
        mymodelstruct.n_nodes = {21,15,4}; //{10*4,8*4,7*4}

        // Defining Inputs
        mymodelstruct.inputcolumns.push_back(0); //
        //mymodelstruct.inputcolumns.push_back(1); //
        //mymodelstruct.inputcolumns.push_back(2); //
        mymodelstruct.inputcolumns.push_back(3); //
        //mymodelstruct.inputcolumns.push_back(4); //
        //mymodelstruct.inputcolumns.push_back(5); //
        mymodelstruct.inputcolumns.push_back(6); //
        mymodelstruct.inputcolumns.push_back(7); //
        //mymodelstruct.inputcolumns.push_back(8); //

        // Defining Output(s)
        for (int i = 0; i<number_of_outputs; i++)
            mymodelstruct.outputcolumns.push_back(total_data_cols-(i+1)); // Output: Settling element (1)_Solids:concentration

        // Lags definition
        vector<int> lag0; lag0.push_back(0);
        //vector<int> lag1; lag1.push_back(0);
        //vector<int> lag2; lag2.push_back(0);
        vector<int> lag3; lag3.push_back(0); lag3.push_back(18); lag3.push_back(27); lag3.push_back(36);
        //vector<int> lag4; lag4.push_back(0);
        //vector<int> lag5; lag5.push_back(0);
        vector<int> lag6; lag6.push_back(0); lag6.push_back(9);
        vector<int> lag7; lag7.push_back(9); lag7.push_back(18); lag7.push_back(36);
        //vector<int> lag8; lag8.push_back(0);

        mymodelstruct.lags.push_back(lag0);
        //mymodelstruct.lags.push_back(lag1);
        //mymodelstruct.lags.push_back(lag2);
        mymodelstruct.lags.push_back(lag3);
        //mymodelstruct.lags.push_back(lag4);
        //mymodelstruct.lags.push_back(lag5);
        mymodelstruct.lags.push_back(lag6);
        mymodelstruct.lags.push_back(lag7);
        //mymodelstruct.lags.push_back(lag8);
    }

    }

    else if (!ASM)
    {
    // ------------------------simple model properties for optimized structure----------------------------------------------------
    // Defining Model Structure
    mymodelstruct.dt=0.1;
    mymodelstruct.log_output=log_output_d;
    mymodelstruct.realization=Realization;
    mymodelstruct.seed_number=Seed_number;

    //Network structure
    mymodelstruct.n_layers = 1;
    mymodelstruct.n_nodes = {4};

    // Defining Inputs
    for (int i=0; i<total_data_cols-1; i++)
    {
    mymodelstruct.inputcolumns.push_back(i); // Input 0: Inflow
    }

    // Defining Output(s)
    mymodelstruct.outputcolumns.push_back(total_data_cols-1); // Output: Settling element (1)_Solids:concentration


    // Lags definition
    vector<int> lag0; lag0.push_back(0);lag0.push_back(14);
    vector<int> lag1; lag1.push_back(14);
    vector<int> lag2; lag2.push_back(7);lag2.push_back(28);

    mymodelstruct.lags.push_back(lag0);
    mymodelstruct.lags.push_back(lag1);
    mymodelstruct.lags.push_back(lag2);
    }

    // ---------------------------------------------GA---------------------------------------------------------
    QFile results;
    QTextStream out;

    for (int r=0; r<Realization; r++)
    {

        if (ASM) {
        mymodelstruct.trainaddress.push_back(datapath_ASM + "observedoutput_train_" + data_name + ".txt");
        mymodelstruct.testaddress.push_back(datapath_ASM + "observedoutput_test_" + data_name + ".txt");

        //mymodelstruct.trainaddress.push_back(datapath_ASM + "observedoutput_train_" + to_string(r) + ".txt");
        //mymodelstruct.testaddress.push_back(datapath_ASM + "observedoutput_test_" + to_string(r) + ".txt");

        mymodelstruct.outputpath = path_ASM + "Results/";

        mymodelstruct.trainobservedaddress.push_back(mymodelstruct.outputpath + "TrainOutputDataTS_" + to_string(r) + ".csv");
        mymodelstruct.trainpredictedaddress.push_back(mymodelstruct.outputpath + "TrainDataPredictionTS_" + to_string(r) + ".csv");

        mymodelstruct.testobservedaddress.push_back(mymodelstruct.outputpath + "TestOutputDataTS_" + to_string(r) + ".csv");
        mymodelstruct.testpredictedaddress.push_back(mymodelstruct.outputpath + "TestDataPredictionTS_" + to_string(r) + ".csv");

        }

        else if (!ASM) {
        mymodelstruct.trainaddress.push_back(datapath + "observedoutput_train_" + data_name + ".txt");
        mymodelstruct.testaddress.push_back(datapath + "observedoutput_test_" + data_name + ".txt");

        //mymodelstruct.trainaddress.push_back(datapath + "observedoutput_train_" + to_string(r) + ".txt");
        //mymodelstruct.testaddress.push_back(datapath + "observedoutput_test_" + to_string(r) + ".txt");

        mymodelstruct.outputpath = path + "Results/";

        mymodelstruct.trainobservedaddress.push_back(mymodelstruct.outputpath + "TrainOutputDataTS_" + data_name + ".csv");
        mymodelstruct.trainpredictedaddress.push_back(mymodelstruct.outputpath + "TrainDataPredictionTS_" + data_name + ".csv");

        mymodelstruct.testobservedaddress.push_back(mymodelstruct.outputpath + "TestOutputDataTS_" + data_name + ".csv");
        mymodelstruct.testpredictedaddress.push_back(mymodelstruct.outputpath + "TestDataPredictionTS_" + data_name + ".csv");

        //mymodelstruct.trainobservedaddress.push_back(mymodelstruct.outputpath + "TrainOutputDataTS_" + to_string(r) + ".csv");
        //mymodelstruct.trainpredictedaddress.push_back(mymodelstruct.outputpath + "TrainDataPredictionTS_" + to_string(r) + ".csv");

        //mymodelstruct.testobservedaddress.push_back(mymodelstruct.outputpath + "TestOutputDataTS_" + to_string(r) + ".csv");
        //mymodelstruct.testpredictedaddress.push_back(mymodelstruct.outputpath + "TestDataPredictionTS_" + to_string(r) + ".csv");

        }

    }

    if (GA) {

        GeneticAlgorithm<ModelCreator> GA;
        GA.Settings.generations = GA_Nsim;
        GA.Settings.MSE_optimization = MSE_Test;
        GA.Settings.outputpath = mymodelstruct.outputpath;
        GA.model = modelCreator;
        GA.model.FFN.ModelStructure = mymodelstruct;

        ModelCreator OptimizedModel = GA.Optimize();
        cout<<"Optimized Model Structure: " << OptimizedModel.FFN.ModelStructure.ParametersToString().toStdString()<<endl;
        OptimizedModel.FFN.silent = false;
        OptimizedModel.FFN.DataSave(datacategory::Train);
        OptimizedModel.FFN.DataSave(datacategory::Test);
        //We can test here:

        if (ASM) {
            QFile results(QString::fromStdString(path_ASM) + "modelresults.txt");
            //QTextStream out;
            if (results.open(QIODevice::WriteOnly | QIODevice::Text)) {
                out.setDevice(&results);
            } else {
                // Handle file open error
                qDebug() << "Error opening file!";
                return 0;
            }
        }

        else if(!ASM) {
            QFile results(QString::fromStdString(path) + "modelresults.txt");
            //QTextStream out;
            if (results.open(QIODevice::WriteOnly | QIODevice::Text)) {
                out.setDevice(&results);
            } else {
                // Handle file open error
                qDebug() << "Error opening file!";
                return 0;
            }
        }

    }

    // ---------------------------------------------RMS---------------------------------------------------------

    else if (!GA) {

        if (randommodelstructure) {

            std::ofstream file;
            file.open(datapath_ASM + "Results/RMS_Output.txt", std::ios::out);
            for (int i=0; i<Random_Nsim; i++) // Random Model Structure Generation
            {

                modelCreator.CreateRandomModelStructure(&mymodelstruct);

                // Running FFNWrapper
                if (mymodelstruct.ValidLags())
                {   FFNWrapper_Multi F;
                    F.silent = false;
                    F.ModelStructure = mymodelstruct;
                    F.Initiate();
                    F.Train();
                    F.Test();
                    F.PerformanceMetrics();

                    F.DataSave(datacategory::Train);
                    F.DataSave(datacategory::Test);
                    //F.Plotter();
                    //F.Optimizer();

                    file<<F.ModelStructure.ParametersToString().toStdString()<<endl;

                    //data::Save("model.xml","model", F);
                }
                else
                    i--;
            }
            file.close();

        }

    // ---------------------------------------------Optimized model---------------------------------------------------------

        else if (!randommodelstructure) {

            FFNWrapper_Multi F;
            F.silent = false;
            F.ModelStructure = mymodelstruct;
            F.Initiate();
            F.Train();
            F.Test();
            F.PerformanceMetrics();

            //F.silent = false;
            F.DataSave(datacategory::Train);
            F.DataSave(datacategory::Test);
            F.Plotter();
            //F.Optimizer();

            //data::Save("model.xml","model", F);

            /*
            FFNWrapper_Multi F1;
            F1.silent = false;
            F1.ModelStructure = mymodelstruct;
            F1.Initiate();
            F1.Train();
            F1.Test();
            F1.PerformanceMetrics();

            //F1.silent = false;
            F1.DataSave(datacategory::Train);
            F1.DataSave(datacategory::Test);
            F1.Plotter();
            //F1.Optimizer();
            */

        }

        else
            cout << "No estimation implemented!";

    }

    return 0;
}

