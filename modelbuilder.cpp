/**
 * @file modelbuilder.cpp
 * @brief Implements the function that builds the full model structure for FFN multi-output networks.
 *
 * @details
 * This file contains one of the essential components of the FFN Wrapper:
 * the construction of the internal neural network structure (layers, nodes, inputs, outputs, lags).
 *
 * The function @ref BuildModelStructure():
 *
 * - Uses the @c Config struct to determine model type (ASM vs Settling)
 * - Creates constituent-specific architectures for ASM (NO, NH, sCOD, TKN, VSS, ND)
 * - Assigns input columns according to optimized GA designs
 * - Assigns output columns automatically based on total columns and number of outputs
 * - Creates lag vectors for each input column
 * - Sets model parameters such as dt, seed_number, realization flags
 *
 * The goal is to keep main.cpp clean, while preserving all original logic from the monolithic code.
 *
 * @note
 *   The architecture definitions in this file come from previously optimized GA experiments.
 *
 * @see BuildAddresses()
 * @see CModelStructure_Multi
 * @see Config
 */

#include "modelbuilder.h"

/**
 * @brief Build the full FFN model structure (layers, nodes, inputs/outputs, lags).
 *
 * @details
 * This function populates all required fields in @c CModelStructure_Multi:
 *
 * - **Network Architecture**
 *   - `n_layers`: number of hidden layers
 *   - `n_nodes`: vector of node counts per hidden layer
 *
 * - **Input Columns**
 *   - Selected based on constituent type (optimized manually/GA-validated)
 *
 * - **Output Columns**
 *   - Always assigned as the last `number_of_outputs` columns
 *
 * - **Lag Structure**
 *   - Each input column has a list of lag offsets (in time steps)
 *   - Different constituents have different lag patterns
 *
 * - **General Model Settings**
 *   - GA flag
 *   - Seed
 *   - dt
 *   - Log transform flag
 *   - Number of realizations
 *
 * ## MODEL TYPES
 * ### 1. Settling Model (Non-ASM)
 * Simple model with:
 * - 1 hidden layer
 * - Fixed lags {0,14}, {14}, {7,28}
 *
 * ### 2. ASM Model
 * Supported constituents:
 * - NO
 * - NH
 * - sCOD
 * - TKN
 * - VSS
 * - ND
 *
 * Each is assigned:
 * - Optimized number of layers
 * - Optimized number of nodes
 * - A hand-crafted input selection
 * - A hand-crafted lag definition
 *
 * @param ms  Reference to the model structure object to populate.
 * @param cfg Configuration structure containing all model settings.
 */
void BuildModelStructure(CModelStructure_Multi& ms, const Config& cfg)
{
    // ========================================================================
    // 1. General model-independent settings
    // ========================================================================
    ms.GA           = cfg.GA_switch;
    ms.dt           = 0.1;
    ms.log_output   = cfg.log_output_d;
    ms.realization  = cfg.Realization;
    ms.seed_number  = cfg.Seed_number;

    const std::string& data_name = cfg.data_name;
    double total_data_cols       = cfg.total_data_cols;
    double number_of_outputs     = cfg.number_of_outputs;

    // ========================================================================
    // 2. SIMPLE NON-ASM (Settling) MODEL
    // ========================================================================
    if (!cfg.ASM)
    {
        /**
         * Simple model logic:
         * - Only one hidden layer
         * - All inputs are used except the final output column
         * - Very limited lag structure
         */
        ms.n_layers = 1;
        ms.n_nodes  = {4};

        // Use all input columns except the last one (output)
        for (int i = 0; i < total_data_cols - 1; i++)
            ms.inputcolumns.push_back(i);

        // Output column is always last
        ms.outputcolumns.push_back(total_data_cols - 1);

        // Hard-coded simple lag structure
        ms.lags = {
            {0,14},   // input 0
            {14},     // input 1
            {7,28}    // input 2
        };

        return;
    }

    // ========================================================================
    // 3. ASM MODEL — Constituent-Specific Architectures
    // ========================================================================
    // Every ASM structure below is a direct translation of original manual GA-optimized rules.

    // ------------------------------------------------------------------------
    // Constituent: NO (Nitrate)
    // ------------------------------------------------------------------------
    if (data_name == "NO")
    {
        ms.n_layers = 2;
        ms.n_nodes  = {39,17};

        ms.inputcolumns = {0,1,2,3,6,7,8};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {0,22,33},  // col 0
            {22,33},    // col 1 (WAS)
            {44},       // col 2
            {0,33,44},  // col 3
            {0,11,22},  // col 6
            {44},       // col 7
            {11,22}     // col 8
        };

        return;
    }

    // ------------------------------------------------------------------------
    // Constituent: NH (Ammonia)
    // ------------------------------------------------------------------------
    if (data_name == "NH")
    {
        ms.n_layers = 3;
        ms.n_nodes  = {19,8,4};

        ms.inputcolumns = {0,1,3,4,6,7};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {1},        // col 0
            {2,3},      // col 1 (WAS)
            {1,3},      // col 3
            {2},        // col 4
            {0,2},      // col 6
            {1}         // col 7
        };

        return;
    }

    // ------------------------------------------------------------------------
    // Constituent: sCOD (Soluble COD)
    // ------------------------------------------------------------------------
    if (data_name == "sCOD")
    {
        ms.n_layers = 3;
        ms.n_nodes  = {36,37,7};

        ms.inputcolumns = {0,1,2,3,5,6,7,8};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {0,1,3},          // col 0
            {0,1,2,4},        // col 1 (WAS)
            {0,1,2,4},        // col 2
            {0,2},            // col 3
            {0,1,2},          // col 5
            {2},              // col 6
            {0,1,3},          // col 7
            {0,2,3}           // col 8
        };

        return;
    }

    // ------------------------------------------------------------------------
    // Constituent: TKN (Total Kjeldahl Nitrogen)
    // ------------------------------------------------------------------------
    if (data_name == "TKN")
    {
        ms.n_layers = 3;
        ms.n_nodes  = {11,8,9};

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

        return;
    }

    // ------------------------------------------------------------------------
    // Constituent: VSS (Volatile Suspended Solids)
    // ------------------------------------------------------------------------
    if (data_name == "VSS")
    {
        ms.n_layers = 3;
        ms.n_nodes  = {11,5,2};

        ms.inputcolumns = {0,7};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {0,12,60},   // col 0
            {0,48}       // col 7
        };

        return;
    }

    // ------------------------------------------------------------------------
    // Constituent: ND (Nitrite + Nitrate Dynamics)
    // ------------------------------------------------------------------------
    if (data_name == "ND")
    {
        ms.n_layers = 3;
        ms.n_nodes  = {21,15,4};

        ms.inputcolumns = {0,3,6,7};

        for (int i = 0; i < number_of_outputs; i++)
            ms.outputcolumns.push_back(total_data_cols - (i+1));

        ms.lags = {
            {0},                // col 0
            {0,18,27,36},       // col 3
            {0,9},              // col 6
            {9,18,36}           // col 7
        };

        return;
    }
}
