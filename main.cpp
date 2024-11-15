//Our main

#include <iostream>
#include <mlpack.hpp>
#include "ffnwrapper.h"
#include <BTCSet.h>
#include "modelcreator.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>

using namespace mlpack;
using namespace std;


int main()
{

    string path;
#ifdef Arash
    path = "/home/arash/Projects/FFNWrapper/";
#else
    path = "/home/behzad/Projects/FFNWrapper2/";
#endif

    // Defining Model Structure
    CModelStructure mymodelstruct;
    mymodelstruct.n_layers = 1;
    mymodelstruct.n_nodes = {2};

    mymodelstruct.dt=0.01;
    string datapath = "/home/behzad/Projects/Settling_Models/";
    string buildpath = "build/Desktop_Qt_5_15_2_GCC_64bit-Debug/";

<<<<<<< HEAD
    int randommodelstructure = 0; // 0 for no random model structure usage and 1 for random model structure usage

    for (int r=0; r<5; r++) // Realization
=======
    for (int r=0; r<3; r++) // Realization
>>>>>>> parent of 6474ccb (11.15 Reazlization (Multi))

    {
        mymodelstruct.realization = r;

        mymodelstruct.inputaddress = datapath + "observedoutput_" + to_string(r) + ".txt";
        mymodelstruct.testaddress = datapath + "observedoutput_" + to_string(r) + ".txt";

        mymodelstruct.outputpath = "/home/behzad/Projects/FFNWrapper2/Results/";
        mymodelstruct.observedaddress = mymodelstruct.outputpath + "TestOutputDataTS_" + to_string(r) + ".csv";
        mymodelstruct.predictedaddress = mymodelstruct.outputpath + "PredictionTS_" + to_string(r) + ".csv";

    // Defining Output(s)
    mymodelstruct.outputcolumns.push_back(3); // Output: V(11): Settling element (1)_Solids:concentration

    //Model creator
    ModelCreator modelCreator;
    modelCreator.lag_frequency = 3;
    modelCreator.maximum_superficial_lag = 5;
    modelCreator.total_number_of_columns = 2;
    modelCreator.max_number_of_layers = 2;
    modelCreator.max_lag_multiplier = 10;

    QFile results(QString::fromStdString(path) + "modelresults.txt");
    QTextStream out;
    if (results.open(QIODevice::WriteOnly | QIODevice::Text)) {
        out.setDevice(&results);
    } else {
        // Handle file open error
        qDebug() << "Error opening file!";
        return 0;
    }

    for (int i=0; i<10; i++) // Random Model Structure Generation

    {

        modelCreator.CreateRandomModelStructure(&mymodelstruct);

        // Running FFNWrapper
        if (mymodelstruct.ValidLags())
        {   FFNWrapper F;
            F.ModelStructure = mymodelstruct;
            F.Initiate();
            F.Training();
            F.Testing();
            F.PerformanceMetrics();

            qDebug()<< "i = " << i << ", " << mymodelstruct.ParametersToString() << ", nMSE = " << F.nMSE << ", R2 = " << F._R2;
            out << "i = " << i << ", " << mymodelstruct.ParametersToString() << ", nMSE = " << F.nMSE << ", R2 = " << F._R2 << "\n";

            F.DataSave();
            F.Plotter();
            //F.Optimizer();

            //data::Save("model.xml","model", F);
        }
        else
            i--;
    }

    results.close();

<<<<<<< HEAD
    }

    else if (randommodelstructure == 0) {

        // Defining Inputs
        mymodelstruct.inputcolumns.push_back(0); // Input 1: D(2): Settling element (1)_Coagulant:external_mass_flow_timeseries
        mymodelstruct.inputcolumns.push_back(1); // Input 2: CV(50): Reactor (1)_Solids:inflow_concentration

        //Lags definition
        vector<int> lag1; lag1.push_back(28); //lag1.push_back(20); lag1.push_back(50);
        vector<int> lag2; lag2.push_back(14); //lag2.push_back(10); lag2.push_back(30);
        mymodelstruct.lags.push_back(lag1);
        mymodelstruct.lags.push_back(lag2);


        FFNWrapper F;
        F.ModelStructure = mymodelstruct;
        F.Initiate();
        F.Training();
        F.Testing();
        F.PerformanceMetrics();

        qDebug()<< mymodelstruct.ParametersToString() << ", nMSE = " << F.nMSE << ", R2 = " << F._R2;
        out << mymodelstruct.ParametersToString() << ", nMSE = " << F.nMSE << ", R2 = " << F._R2 << "\n";

        F.DataSave();
        F.Plotter();
        //F.Optimizer();

        //data::Save("model.xml","model", F);

        results.close();
    }

    else
        cout << "No estimation implemented!";

    }
=======
    };
>>>>>>> parent of 6474ccb (11.15 Reazlization (Multi))

    return 0;
}





//mymodelstruct.inputaddress = datapath + "observedoutput.txt";
//mymodelstruct.testaddress = datapath + "observedoutput.txt";

//mymodelstruct.observedaddress = path + buildpath + "TestOutputDataTS.csv";
//mymodelstruct.predictaddress = path + buildpath + "PredictionTS.csv";
/*

    // Defining Inputs
    mymodelstruct.inputcolumns.push_back(0); // Input 1: D(2): Settling element (1)_Coagulant:external_mass_flow_timeseries
    mymodelstruct.inputcolumns.push_back(1); // Input 2: CV(50): Reactor (1)_Solids:inflow_concentration

    //Lags definition
    vector<int> lag1; lag1.push_back(28); //lag1.push_back(20); lag1.push_back(50);
    vector<int> lag2; lag2.push_back(14); //lag2.push_back(10); lag2.push_back(30);
    mymodelstruct.lags.push_back(lag1);
    mymodelstruct.lags.push_back(lag2);

    */

/*

    FFNWrapper F;
    F.ModelStructure = mymodelstruct;
    F.Initiate();
    F.Training();
    F.Testing();
    F.PerformanceMetrics();
    F.DataSave();
    F.Plotter();
    F.Optimizer();

    */


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
    F2.Training();
    F2.Testing();
    F2.PerformanceMetrics();
    F2.DataSave();
    F2.Plotter();
    F2.Optimizer();

    */
