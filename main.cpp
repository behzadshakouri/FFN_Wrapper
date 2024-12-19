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

using namespace mlpack;
using namespace std;


int main()
{

    const double Realization = 2; // Number of Realizations
    const double total_data_cols = 3; // Number of Inputs + Outputs

    bool ASM = true; // true for ASM and false for Settling element simple model

    bool GA = true;  // true for Genetic Alghorithm usage and false for no Genetic Alghorithm usage
    const double GA_Nsim = 1000; // Number of GA simulations ???

    bool randommodelstructure = false; // true for random model structure usage and false for no random model structure usage
    const double Random_Nsim = 1; // Number of random model structure simulations

    //Model creator (Random model structure)
    ModelCreator modelCreator;
    modelCreator.lag_frequency = 3;
    modelCreator.maximum_superficial_lag = 5;
    modelCreator.total_number_of_columns = total_data_cols;
    modelCreator.max_number_of_layers = 3;
    modelCreator.max_lag_multiplier = 10;
    modelCreator.max_number_of_nodes_in_layers = 10;


    string path;
    string path_ASM;

#ifdef Arash
    path = "/home/arash/Projects/FFNWrapper/";
    path_ASM = "/home/arash/Projects/FFNWrapper/ASM/";
    string datapath = "/home/arash/Projects/FFNWrapper/";
    string datapath_ASM = "/home/arash/Projects/FFNWrapper/";
    string buildpath = "build/Desktop_Qt_5_15_2_GCC_64bit-Debug/";
#else
    path = "/home/behzad/Projects/FFNWrapper2/";
    path_ASM = "/home/behzad/Projects/FFNWrapper/ASM/";
    string datapath = "/home/behzad/Projects/FFNWrapper2/";
    string datapath_ASM = "/home/behzad/Projects/FFNWrapper2/ASM/";
    string buildpath = "build/Desktop_Qt_5_15_2_GCC_64bit-Debug/";
#endif

    // Defining Model Structure
    CModelStructure_Multi mymodelstruct; //randommodelstructure
    mymodelstruct.n_layers = 1;
    mymodelstruct.n_nodes = {4};

    mymodelstruct.dt=0.01;

    // Defining Inputs
    mymodelstruct.inputcolumns.push_back(0); // Input 0: Inflow
    mymodelstruct.inputcolumns.push_back(1); // Input 1: Settling element (1)_Coagulant:external_mass_flow_timeseries
    mymodelstruct.inputcolumns.push_back(2); // Input 2: Reactor (1)_Solids:inflow_concentration

    // Defining Output(s)
    mymodelstruct.outputcolumns.push_back(3); // Output: Settling element (1)_Solids:concentration

    //Lags definition
    vector<int> lag0; lag0.push_back(0);lag0.push_back(14);
    vector<int> lag1; lag1.push_back(14);
    vector<int> lag2; lag2.push_back(7);lag2.push_back(28);

    mymodelstruct.lags.push_back(lag0);
    mymodelstruct.lags.push_back(lag1);
    mymodelstruct.lags.push_back(lag2);

    QFile results;
    QTextStream out;

    for (int r=0; r<Realization; r++)
    {

        if (ASM) {
        mymodelstruct.inputaddress.push_back(datapath_ASM + "observedoutput_" + to_string(r) + ".txt");
        mymodelstruct.testaddress.push_back(datapath_ASM + "observedoutput_" + to_string(r) + ".txt");

        mymodelstruct.outputpath = path_ASM + "Results/";
        mymodelstruct.observedaddress.push_back(mymodelstruct.outputpath + "TestOutputDataTS_" + to_string(r) + ".csv");
        mymodelstruct.predictedaddress.push_back(mymodelstruct.outputpath + "PredictionTS_" + to_string(r) + ".csv");
        }

        else if (!ASM) {
        mymodelstruct.inputaddress.push_back(datapath + "observedoutput_" + to_string(r) + ".txt");
        mymodelstruct.testaddress.push_back(datapath + "observedoutput_" + to_string(r) + ".txt");

        mymodelstruct.outputpath = path + "Results/";
        mymodelstruct.observedaddress.push_back(mymodelstruct.outputpath + "TestOutputDataTS_" + to_string(r) + ".csv");
        mymodelstruct.predictedaddress.push_back(mymodelstruct.outputpath + "PredictionTS_" + to_string(r) + ".csv");
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
                    F.ModelStructure = mymodelstruct;
                    F.Initiate();
                    F.Train();
                    F.Test();
                    F.PerformanceMetrics();

                    qDebug()<< "i = " << i << ", " << mymodelstruct.ParametersToString() << ", nMSE = " << F.nMSE << ", R2 = " << F._R2;
                    out << "i = " << i << ", " << mymodelstruct.ParametersToString() << ", nMSE = " << F.nMSE << ", R2 = " << F._R2 << "\n";

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
            //F.silent = false;
            F.ModelStructure = mymodelstruct;
            F.Initiate();
            F.Train();
            F.Test();
            F.PerformanceMetrics();

            qDebug()<< mymodelstruct.ParametersToString() << ", nMSE = " << F.nMSE << ", R2 = " << F._R2;
            out << mymodelstruct.ParametersToString() << ", nMSE = " << F.nMSE << ", R2 = " << F._R2 << "\n";

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








//mymodelstruct.inputaddress = datapath + "observedoutput.txt";
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

    mymodelstruct2.inputaddress = path + "observedoutput.txt";
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
