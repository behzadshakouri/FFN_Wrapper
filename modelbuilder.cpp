#include "modelbuilder.h"

void BuildModelStructure(CModelStructure_Multi& ms, const Config& cfg)
{
    ms.GA = cfg.GA_switch;
    ms.dt = 0.1;
    ms.log_output = cfg.log_output_d;
    ms.realization = cfg.Realization;
    ms.seed_number = cfg.Seed_number;

    const std::string& data_name = cfg.data_name;
    double total_data_cols = cfg.total_data_cols;
    double number_of_outputs = cfg.number_of_outputs;

    // ---------------- SIMPLE NON-ASM MODEL ---------------------
    if (!cfg.ASM)
    {
        ms.n_layers = 1;
        ms.n_nodes = {4};

        for (int i = 0; i < total_data_cols - 1; i++)
            ms.inputcolumns.push_back(i);

        ms.outputcolumns.push_back(total_data_cols - 1);

        ms.lags = {
            {0,14},
            {14},
            {7,28}
        };
        return;
    }

    // ---------------- ASM CASES -------------------------------

    if (data_name == "NO")
    {
        ms.n_layers = 2;
        ms.n_nodes = {39,17};

        ms.inputcolumns = {0,1,2,3,6,7,8};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {0,22,33},
            {22,33},
            {44},
            {0,33,44},
            {0,11,22},
            {44},
            {11,22}
        };
    }
    else if (data_name == "NH")
    {
        ms.n_layers = 3;
        ms.n_nodes = {19,8,4};

        ms.inputcolumns = {0,1,3,4,6,7};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {1},
            {2,3},
            {1,3},
            {2},
            {0,2},
            {1}
        };
    }
    else if (data_name == "sCOD")
    {
        ms.n_layers = 3;
        ms.n_nodes = {36,37,7};

        ms.inputcolumns = {0,1,2,3,5,6,7,8};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {0,1,3},
            {0,1,2,4},
            {0,1,2,4},
            {0,2},
            {0,1,2},
            {2},
            {0,1,3},
            {0,2,3}
        };
    }
    else if (data_name == "TKN")
    {
        ms.n_layers = 3;
        ms.n_nodes = {11,8,9};

        ms.inputcolumns = {0,1,3,5,6,7,8};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {1},
            {0,2},
            {1,2,3},
            {0},
            {0,3,4},
            {4},
            {0,4}
        };
    }
    else if (data_name == "VSS")
    {
        ms.n_layers = 3;
        ms.n_nodes = {11,5,2};

        ms.inputcolumns = {0,7};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {0,12,60},
            {0,48}
        };
    }
    else if (data_name == "ND")
    {
        ms.n_layers = 3;
        ms.n_nodes = {21,15,4};

        ms.inputcolumns = {0,3,6,7};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {0},
            {0,18,27,36},
            {0,9},
            {9,18,36}
        };
    }
}
