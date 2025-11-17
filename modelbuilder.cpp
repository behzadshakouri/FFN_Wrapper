/**
 * @file modelbuilder.cpp
 * @brief Implements the function that builds the full model structure for FFN multi-output networks.
 *
 * @details
 * This file contains the construction of the internal neural network structure
 * (layers, nodes, inputs, outputs, lags) used by FFNWrapper_Multi.
 *
 * It supports:
 * - Simple settling model (non-ASM)
 * - ASM constituents: NO, NH, sCOD, TKN, VSS, ND
 * - Two architecture sets for some constituents:
 *      Config::architecture_set = 0 → original architectures (old GA-optimized)
 *      Config::architecture_set = 1 → new architectures (Behzad 2025 combos)
 */

#include "modelbuilder.h"

void BuildModelStructure(CModelStructure_Multi& ms, const Config& cfg)
{
    // General model settings
    ms.GA           = cfg.GA_switch;
    ms.dt           = 0.1;
    ms.log_output   = cfg.log_output_d;
    ms.realization  = cfg.Realization;
    ms.seed_number  = cfg.Seed_number;

    const std::string& data_name    = cfg.data_name;
    double total_data_cols          = cfg.total_data_cols;
    double number_of_outputs        = cfg.number_of_outputs;
    const int As                    = cfg.architecture_set;  // 0 = old, 1 = new

    // =========================================================================
    // NON-ASM MODEL (Settling)
    // =========================================================================
    if (!cfg.ASM)
    {
        ms.n_layers = 1;
        ms.n_nodes  = {4};

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

    // =========================================================================
    // ASM MODE — Constituent-Specific Architectures
    // =========================================================================

    // ---------------------------------------------------------------------
    // NO
    // ---------------------------------------------------------------------
    if (data_name == "NO")
    {
        // New architecture set (Behzad 2025)
        if (As == 1)
        {
            ms.n_layers = 3;
            ms.n_nodes  = {10,28,2};

            ms.inputcolumns = {0,1,2,3,6,7};
            ms.outputcolumns.push_back(total_data_cols - 1);

            ms.lags = {
                {0,13,39},
                {13,39},
                {0,26,39,52},
                {0,52},
                {0,26,52},
                {0,39,52}
            };
            return;
        }

        // Original architecture set (old)
        ms.n_layers = 2;
        ms.n_nodes  = {39,17};

        ms.inputcolumns = {0,1,2,3,6,7,8};
        ms.outputcolumns.push_back(total_data_cols - 1);

        ms.lags = {
            {0,22,33},
            {22,33},
            {44},
            {0,33,44},
            {0,11,22},
            {44},
            {11,22}
        };
        return;
    }

    // ---------------------------------------------------------------------
    // NH
    // ---------------------------------------------------------------------
    if (data_name == "NH")
    {
        // New architecture set (Behzad 2025)
        if (As == 1)
        {
            ms.n_layers = 3;
            ms.n_nodes  = {23,26,5};

            ms.inputcolumns = {0,1,2,3,6,7,8};
            ms.outputcolumns.push_back(total_data_cols - 1);

            ms.lags = {
                {0,13},
                {0},
                {0,26,39},
                {0},
                {0},
                {26,39},
                {52}
            };
            return;
        }

        // Original architecture
        ms.n_layers = 3;
        ms.n_nodes  = {19,8,4};

        ms.inputcolumns = {0,1,3,4,6,7};
        ms.outputcolumns.push_back(total_data_cols - 1);

        ms.lags = {
            {1},
            {2,3},
            {1,3},
            {2},
            {0,2},
            {1}
        };
        return;
    }

    // ---------------------------------------------------------------------
    // sCOD
    // ---------------------------------------------------------------------
    if (data_name == "sCOD")
    {
        // New architecture set (Behzad 2025)
        if (As == 1)
        {
            ms.n_layers = 2;
            ms.n_nodes  = {37,33};

            ms.inputcolumns = {0,1,2,3,4,6,8};
            ms.outputcolumns.push_back(total_data_cols - 1);

            ms.lags = {
                {0,5},
                {5,20},
                {5,25},
                {5},
                {0},
                {0,15},
                {0}
            };
            return;
        }

        // Original architecture
        ms.n_layers = 3;
        ms.n_nodes  = {36,37,7};

        ms.inputcolumns = {0,1,2,3,5,6,7,8};
        ms.outputcolumns.push_back(total_data_cols - 1);

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
        return;
    }

    // ---------------------------------------------------------------------
    // TKN
    // ---------------------------------------------------------------------
    if (data_name == "TKN")
    {
        // New architecture set (Behzad 2025)
        if (As == 1)
        {
            ms.n_layers = 3;
            ms.n_nodes  = {26,28,7};

            ms.inputcolumns = {0,1,2,3,4,5,6,7};
            ms.outputcolumns.push_back(total_data_cols - 1);

            ms.lags = {
                {0},
                {0},
                {0},
                {2,3},
                {0,1},
                {0},
                {2},
                {0,1,2,3}
            };
            return;
        }

        // Original architecture
        ms.n_layers = 3;
        ms.n_nodes  = {11,8,9};

        ms.inputcolumns = {0,1,3,5,6,7,8};
        ms.outputcolumns.push_back(total_data_cols - 1);

        ms.lags = {
            {1},
            {0,2},
            {1,2,3},
            {0},
            {0,3,4},
            {4},
            {0,4}
        };
        return;
    }

    // ---------------------------------------------------------------------
    // VSS (only one set)
    // ---------------------------------------------------------------------
    if (data_name == "VSS")
    {
        ms.n_layers = 3;
        ms.n_nodes  = {11,5,2};

        ms.inputcolumns = {0,7};
        ms.outputcolumns.push_back(total_data_cols - 1);

        ms.lags = {
            {0,12,60},
            {0,48}
        };
        return;
    }

    // ---------------------------------------------------------------------
    // ND (only one set)
    // ---------------------------------------------------------------------
    if (data_name == "ND")
    {
        ms.n_layers = 3;
        ms.n_nodes  = {21,15,4};

        ms.inputcolumns = {0,3,6,7};
        ms.outputcolumns.push_back(total_data_cols - 1);

        ms.lags = {
            {0},
            {0,18,27,36},
            {0,9},
            {9,18,36}
        };
        return;
    }
}
