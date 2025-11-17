#include "config.h"

void BuildAddresses(CModelStructure_Multi& ms, const Config& cfg)
{
    for (int r = 0; r < cfg.Realization; r++)
    {
        if (cfg.ASM)
        {
            ms.trainaddress.push_back(
                cfg.datapath_ASM + "observedoutput_train_" + cfg.data_name + ".txt");

            ms.testaddress.push_back(
                cfg.datapath_ASM + "observedoutput_test_" + cfg.data_name + ".txt");

            ms.outputpath = cfg.path_ASM + "Results/";

            ms.trainobservedaddress.push_back(
                ms.outputpath + "TrainOutputDataTS_" + std::to_string(r) + ".csv");

            ms.trainpredictedaddress.push_back(
                ms.outputpath + "TrainDataPredictionTS_" + std::to_string(r) + ".csv");

            ms.testobservedaddress.push_back(
                ms.outputpath + "TestOutputDataTS_" + std::to_string(r) + ".csv");

            ms.testpredictedaddress.push_back(
                ms.outputpath + "TestDataPredictionTS_" + std::to_string(r) + ".csv");
        }
        else
        {
            ms.trainaddress.push_back(
                cfg.datapath + "observedoutput_train_" + cfg.data_name + ".txt");

            ms.testaddress.push_back(
                cfg.datapath + "observedoutput_test_" + cfg.data_name + ".txt");

            ms.outputpath = cfg.path + "Results/";

            ms.trainobservedaddress.push_back(
                ms.outputpath + "TrainOutputDataTS_" + cfg.data_name + ".csv");

            ms.trainpredictedaddress.push_back(
                ms.outputpath + "TrainDataPredictionTS_" + cfg.data_name + ".csv");

            ms.testobservedaddress.push_back(
                ms.outputpath + "TestOutputDataTS_" + cfg.data_name + ".csv");

            ms.testpredictedaddress.push_back(
                ms.outputpath + "TestDataPredictionTS_" + cfg.data_name + ".csv");
        }
    }
}
