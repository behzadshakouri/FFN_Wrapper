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

    bool GA = true;  // true for Genetic Alghorithm usage and false for no Genetic Alghorithm usage
    const double GA_Nsim = 100; // Number of GA simulations ???

    bool randommodelstructure = false; // true for random model structure usage and false for no random model structure usage
    const double Random_Nsim = 100; // Number of random model structure simulations

    //------------------------------------------------------------------------------------------------------------------------------

    //Model creator (Random model structure)
    ModelCreator modelCreator;
    modelCreator.lag_frequency = 3;
    modelCreator.maximum_superficial_lag = 5;
    modelCreator.total_number_of_columns = total_data_cols-number_of_outputs; // Inputs
    modelCreator.max_number_of_layers = 4; // 4 to 6
    modelCreator.max_lag_multiplier = 10;
    modelCreator.max_number_of_nodes_in_layers = 10*2; // 10 to 15


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
    mymodelstruct.n_layers = 3;
    mymodelstruct.n_nodes = {10*2,8*2,7*2};

    mymodelstruct.dt=0.1;

    // Defining Inputs

    for (int i=0; i<total_data_cols-number_of_outputs; i++)
    {
        mymodelstruct.inputcolumns.push_back(i); // Input 0: Inflow
    }


    // Defining Output(s)
    for (int i = 0; i<number_of_outputs; i++)
        mymodelstruct.outputcolumns.push_back(total_data_cols-(i+1)); // Output: Settling element (1)_Solids:concentration


    // Lags definition
    vector<int> lag0; lag0.push_back(2); lag0.push_back(4); lag0.push_back(6); lag0.push_back(8);
    vector<int> lag1; lag1.push_back(0); lag1.push_back(2); lag1.push_back(4); lag1.push_back(8);
    vector<int> lag2; lag2.push_back(2);
    vector<int> lag3; lag3.push_back(0); lag3.push_back(6);
    vector<int> lag4; lag4.push_back(8);
    vector<int> lag5; lag5.push_back(6);
    vector<int> lag6; lag6.push_back(2); lag6.push_back(6);
    vector<int> lag7; lag7.push_back(2);
    vector<int> lag8; lag8.push_back(4); lag8.push_back(10);

    mymodelstruct.lags.push_back(lag0);
    mymodelstruct.lags.push_back(lag1);
    mymodelstruct.lags.push_back(lag2);
    mymodelstruct.lags.push_back(lag3);
    mymodelstruct.lags.push_back(lag4);
    mymodelstruct.lags.push_back(lag5);
    mymodelstruct.lags.push_back(lag6);
    mymodelstruct.lags.push_back(lag7);
    mymodelstruct.lags.push_back(lag8);

    /*
    mymodelstruct.inputcolumns.push_back(0);
    mymodelstruct.inputcolumns.push_back(1);
    //mymodelstruct.inputcolumns.push_back(2);
    mymodelstruct.inputcolumns.push_back(3);
    //mymodelstruct.inputcolumns.push_back(4);
    //mymodelstruct.inputcolumns.push_back(5);
    mymodelstruct.inputcolumns.push_back(6);
    //mymodelstruct.inputcolumns.push_back(7);
    //mymodelstruct.inputcolumns.push_back(8);
    */

    /*
    // Lags definition
    vector<int> lag0; lag0.push_back(1); lag0.push_back(2);
    vector<int> lag1; lag1.push_back(1); lag1.push_back(3);
    //vector<int> lag2; lag2.push_back(0);
    vector<int> lag3; lag3.push_back(0); lag3.push_back(2); lag3.push_back(3); lag3.push_back(4);
    //vector<int> lag4; lag4.push_back(0);
    //vector<int> lag5; lag5.push_back(0);
    vector<int> lag6; lag6.push_back(1); lag6.push_back(3); lag6.push_back(4);
    //vector<int> lag7; lag7.push_back(0);
    //vector<int> lag8; lag8.push_back(0);

    // Lags definition
    vector<int> lag0; lag0.push_back(9); lag0.push_back(36); //lag0.push_back(60);
    vector<int> lag1; lag1.push_back(0); lag1.push_back(36); //lag1.push_back(60);
    vector<int> lag2; lag2.push_back(36); //lag2.push_back(60);
    vector<int> lag3; lag3.push_back(36); lag3.push_back(60);
    vector<int> lag4; lag4.push_back(0); lag4.push_back(36); lag4.push_back(60);
    vector<int> lag5; lag5.push_back(0); lag5.push_back(27); lag5.push_back(60);
    vector<int> lag6; lag6.push_back(0); lag6.push_back(27); lag6.push_back(60);
    vector<int> lag7; lag7.push_back(18); lag7.push_back(27); lag7.push_back(36); lag7.push_back(60);

    mymodelstruct.lags.push_back(lag0);
    mymodelstruct.lags.push_back(lag1);
    //mymodelstruct.lags.push_back(lag2);
    mymodelstruct.lags.push_back(lag3);
    //mymodelstruct.lags.push_back(lag4);
    //mymodelstruct.lags.push_back(lag5);
    mymodelstruct.lags.push_back(lag6);
    //mymodelstruct.lags.push_back(lag7);
    //mymodelstruct.lags.push_back(lag8);
    */



    }
    else if (!ASM)
    {
    // ------------------------simple model properties for optimized structure----------------------------------------------------
    // Defining Model Structure
    mymodelstruct.n_layers = 1;
    mymodelstruct.n_nodes = {4};

    mymodelstruct.dt=0.01;

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
        mymodelstruct.trainaddress.push_back(datapath_ASM + "observedoutput_train_" + to_string(r) + ".txt");
        mymodelstruct.testaddress.push_back(datapath_ASM + "observedoutput_test_" + to_string(r) + ".txt");

        mymodelstruct.outputpath = path_ASM + "Results/";

        mymodelstruct.trainobservedaddress.push_back(mymodelstruct.outputpath + "TrainOutputDataTS_" + to_string(r) + ".csv");
        mymodelstruct.trainpredictedaddress.push_back(mymodelstruct.outputpath + "TrainDataPredictionTS_" + to_string(r) + ".csv");

        mymodelstruct.testobservedaddress.push_back(mymodelstruct.outputpath + "TestOutputDataTS_" + to_string(r) + ".csv");
        mymodelstruct.testpredictedaddress.push_back(mymodelstruct.outputpath + "TestDataPredictionTS_" + to_string(r) + ".csv");

        }

        else if (!ASM) {
        mymodelstruct.trainaddress.push_back(datapath + "observedoutput_train_" + to_string(r) + ".txt");
        mymodelstruct.testaddress.push_back(datapath + "observedoutput_test_" + to_string(r) + ".txt");

        mymodelstruct.outputpath = path + "Results/";

        mymodelstruct.trainobservedaddress.push_back(mymodelstruct.outputpath + "TrainOutputDataTS_" + to_string(r) + ".csv");
        mymodelstruct.trainpredictedaddress.push_back(mymodelstruct.outputpath + "TrainDataPredictionTS_" + to_string(r) + ".csv");

        mymodelstruct.testobservedaddress.push_back(mymodelstruct.outputpath + "TestOutputDataTS_" + to_string(r) + ".csv");
        mymodelstruct.testpredictedaddress.push_back(mymodelstruct.outputpath + "TestDataPredictionTS_" + to_string(r) + ".csv");

        }

    }

    if (GA) {

    GeneticAlgorithm<ModelCreator> GA;
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

    else if (!GA) {

        if (randommodelstructure) {
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

                    //qDebug()<< "i = " << i << ", " << mymodelstruct.ParametersToString() << ", nMSE_Train = " << F.nMSE_Train << ", R2_Train = " << F._R2_Train;
                    //out << "i = " << i << ", " << mymodelstruct.ParametersToString() << ", nMSE_Train = " << F.nMSE_Train << ", R2_Train = " << F._R2_Train << "\n";

                    //qDebug()<< "i = " << i << ", " << mymodelstruct.ParametersToString() << ", nMSE_Test = " << F.nMSE_Test << ", R2_Test = " << F._R2_Test;
                    //out << "i = " << i << ", " << mymodelstruct.ParametersToString() << ", nMSE_Test = " << F.nMSE_Test << ", R2_Test = " << F._R2_Test << "\n";

                    F.DataSave(datacategory::Train);
                    F.DataSave(datacategory::Test);
                    //F.Plotter();
                    //F.Optimizer();

                    //data::Save("model.xml","model", F);
                }
                else
                    i--;
            }

        results.close();

        }

        else if (!randommodelstructure) {

            FFNWrapper_Multi F;
            F.silent = false;
            F.ModelStructure = mymodelstruct;
            F.Initiate();
            F.Train();
            F.Test();
            F.PerformanceMetrics();

            //qDebug()<< mymodelstruct.ParametersToString() << ", nMSE_Train = " << F.nMSE_Train << ", R2_Train = " << F._R2_Train;
            //out << mymodelstruct.ParametersToString() << ", nMSE_Train = " << F.nMSE_Train << ", R2_Train = " << F._R2_Train << "\n";

            //qDebug()<< mymodelstruct.ParametersToString() << ", nMSE_Test = " << F.nMSE_Test << ", R2_Test = " << F._R2_Test;
            //out << mymodelstruct.ParametersToString() << ", nMSE_Test = " << F.nMSE_Test << ", R2_Test = " << F._R2_Test << "\n";

            //F.silent = false;
            F.DataSave(datacategory::Train);
            F.DataSave(datacategory::Test);
            F.Plotter();
            //F.Optimizer();

            //data::Save("model.xml","model", F);

            results.close();
        }

        else
            cout << "No estimation implemented!";

    }

    return 0;
}






/*
    //Testing CTransformation class
    CTransformation transformer;
    arma::mat data = {{1, 2, 3,2}, {4, 5, 6,5.5}, {7, 8, 9,7.5} };

    std::cout << "Original Data:\n" << data << std::endl;

    // Normalize data
    arma::mat normalizedData = transformer.normalize(data);
    std::cout << "Normalized Data:\n" << normalizedData << std::endl;

    // Save parameters
    transformer.saveParameters("scaling_params.txt");

    // Load parameters
    CTransformation newTransformer;
    newTransformer.loadParameters("scaling_params.txt");

    // Inverse transform
    arma::mat restoredData = newTransformer.inverseTransform(normalizedData);
    std::cout << "Restored Data:\n" << restoredData << std::endl;

    arma::mat newdata = {{1.2, 2.3, 1.4, 0.8}, {4, 5, 6, 4.5}, {7, 8, 9, 8.6}};
    arma::mat newnormalizedData = newTransformer.transform(newdata);
    std::cout << "Normalized Data:\n" << newnormalizedData << std::endl;
    */



//mymodelstruct.trainaddress = datapath + "observedoutput.txt";
//mymodelstruct.testaddress = datapath + "observedoutput.txt";

//mymodelstruct.observedaddress = path + buildpath + "TestOutputDataTS.csv";
//mymodelstruct.predictaddress = path + buildpath + "PredictionTS.csv";
/*


//CModelStructure mymodelstruct2;
//CModelStructure mymodelstruct3(mymodelstruct);
//modelCreator.CreateRandomModelStructure(&mymodelstruct2);

//qDebug()<<"Model2 ?= Model1"<<(mymodelstruct2==mymodelstruct);
//qDebug()<<"Model3 ?= Model1"<<(mymodelstruct3==mymodelstruct);


/*
     *
    // NEW MODEL

    // Defining Model Structure
    CModelStructure mymodelstruct2;

    mymodelstruct2.n_layers = 1;
    mymodelstruct2.n_nodes = {2};

    mymodelstruct2.dt=0.01;

    mymodelstruct2.trainaddress = path + "observedoutput.txt";
    mymodelstruct2.testaddress = path + "observedoutput.txt";

    // Defining Inputs
    mymodelstruct2.inputcolumns.push_back(0); // Input 1: D(2): Settling element (1)_Coagulant:external_mass_flow_timeseries
    mymodelstruct2.inputcolumns.push_back(1); // Input 2: CV(50): Reactor (1)_Solids:inflow_concentration

    //Lags definition
    vector<int> lag11; lag11.push_back(28); //lag1.push_back(20); lag1.push_back(50);
    vector<int> lag22; lag22.push_back(1); //lag2.push_back(10); lag2.push_back(30);
    mymodelstruct2.lags.push_back(lag11);
    mymodelstruct2.lags.push_back(lag22);

    // Defining Output(s)
    mymodelstruct2.outputcolumns.push_back(2); // Output: V(11): Settling element (1)_Solids:concentration

    FFNWrapper F2;
    F2.ModelStructure = mymodelstruct2;
    F2.Initiate();
    F2.Train();
    F2.Test();
    F2.PerformanceMetrics();
    F2.DataSave();
    F2.Plotter();
    F2.Optimizer();

    */
