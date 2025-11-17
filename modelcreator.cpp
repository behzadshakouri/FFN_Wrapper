/**
 * @file modelcreator.cpp
 * @brief Implements ModelCreator, the engine responsible for generating, mutating,
 *        decoding, and evaluating model structures for the FFN Wrapper.
 *
 * @details
 * ModelCreator handles:
 * - Random model structure generation for RMS mode
 * - GA chromosome decoding/encoding
 * - Node/layer/lag selection from encoded integers
 * - Mapping base-N integers to NN architecture features
 * - Fitness evaluation using FFNWrapper_Multi
 *
 * Its companion header @ref modelcreator.h documents the API surface.
 */

#include "modelcreator.h"
#include <QFile>
#include <QTextStream>
#include <gsl/gsl_rng.h>
#include <BTCSet.h>

// ======================================================================
//  Constructor
// ======================================================================

/**
 * @brief Default constructor initializes GSL random number generator.
 *
 * @details
 * Seeds RNG using UNIX time. Required for random model structure generation.
 *
 * @note Wrapped in `#ifdef GSL` for optional compilation.
 */
ModelCreator::ModelCreator()
{
#ifdef GSL
    const gsl_rng_type *A = gsl_rng_default;
    r = gsl_rng_alloc(A);

    unsigned long seed = static_cast<unsigned long>(std::time(nullptr));
    gsl_rng_set(r, seed);
#endif
}

// ======================================================================
//  Structure clearing utilities
// ======================================================================

/**
 * @brief Reset all variable-length arrays inside a CModelStructure object.
 *
 * @param modelstructure Pointer to structure to clear.
 *
 * @note Leaves basic metadata untouched.
 */
void ModelCreator::clear(CModelStructure *modelstructure)
{
    modelstructure->lags.clear();
    modelstructure->inputcolumns.clear();
    modelstructure->n_nodes.clear();
}

/**
 * @brief Reset all variable-length arrays inside a CModelStructure_Multi object.
 *
 * @param modelstructure Pointer to structure to clear.
 */
void ModelCreator::clear(CModelStructure_Multi *modelstructure)
{
    modelstructure->lags.clear();
    modelstructure->inputcolumns.clear();
    modelstructure->n_nodes.clear();
}


// ======================================================================
//  Random model structure generation (single-output)
// ======================================================================

/**
 * @brief Create a random model structure for single-output FFN.
 *
 * @details
 * Encoded chromosome:
 * - parameters[0] → Column-selection mask (base 2)
 * - parameters[1] → Lag multiplier
 * - parameters[2 .. 2+Ncols-1] → Lag activation masks
 * - parameters[last] → Node/layer selection (base = max nodes)
 *
 * Random sampling uses GSL RNG.
 *
 * @param modelstructure Output pointer where decoded structure is written.
 * @return true Always true unless total_number_of_columns is zero.
 */
bool ModelCreator::CreateRandomModelStructure(CModelStructure *modelstructure)
{
    long unsigned int max_column_selection = pow(2, total_number_of_columns);
    long unsigned int max_lag_selection    = pow(lag_frequency, maximum_superficial_lag);
    long unsigned int max_node_selection   = pow(max_number_of_layers, max_number_of_layers + 1) - 1;

    parameters.resize(ParametersSize());

    parameters[0] = gsl_rng_uniform_int(r, max_column_selection - 1) + 1;
    parameters[1] = gsl_rng_uniform_int(r, max_lag_multiplier - 1) + 1;

    for (int i = 0; i < total_number_of_columns; i++)
        parameters[i + 2] = gsl_rng_uniform_int(r, max_lag_selection - 1) + 1;

    parameters[total_number_of_columns + 2] =
        gsl_rng_uniform_int(r, max_node_selection - 1) + 1;

    clear(modelstructure);
    CreateModel(modelstructure);
    return true;
}


// ======================================================================
//  Parameter bounds for GA
// ======================================================================

/**
 * @brief Compute maximum allowed integer value for ith evolutionary parameter.
 *
 * @param i Parameter index.
 * @return Maximum possible value (inclusive).
 *
 * @note Used by GA to define search range for each chromosome gene.
 */
long unsigned int ModelCreator::MaxParameter(int i)
{
    if (i == 0) return pow(2, total_number_of_columns) - 1;
    if (i == 1) return max_lag_multiplier - 1;
    if (i < total_number_of_columns + 2) return pow(lag_frequency, maximum_superficial_lag) - 1;
    if (i == total_number_of_columns + 2) return pow(max_number_of_layers, max_number_of_layers + 1) - 1;
    return 0;
}


// ======================================================================
//  Assign GA chromosome to internal parameter vector
// ======================================================================

/**
 * @brief Store GA chromosome values into internal parameter vector.
 *
 * @param x Chromosome vector (unsigned ints).
 *
 * @note Adds +1 because internal encoding begins at 1 instead of 0.
 */
void ModelCreator::AssignParameters(const vector<long unsigned int> &x)
{
    if (x.size() != total_number_of_columns + 3) return;

    parameters.resize(ParametersSize());
    for (unsigned int i = 0; i < total_number_of_columns + 3; i++)
        parameters[i] = x[i] + 1;
}


// ======================================================================
//  Random model structure generation (multi-output)
// ======================================================================

/**
 * @brief Create a random multi-output model structure.
 *
 * @details
 * Identical to single-output version except applied to CModelStructure_Multi.
 *
 * @param modelstructure Output container.
 * @return true Always true.
 */
bool ModelCreator::CreateRandomModelStructure(CModelStructure_Multi *modelstructure)
{
    long unsigned int max_column_selection = pow(2, total_number_of_columns);
    long unsigned int max_lag_selection    = pow(lag_frequency, maximum_superficial_lag);
    long unsigned int max_node_selection   = pow(max_number_of_layers, max_number_of_layers + 1) - 1;

    parameters.resize(ParametersSize());

    parameters[0] = gsl_rng_uniform_int(r, max_column_selection - 1) + 1;
    parameters[1] = gsl_rng_uniform_int(r, max_lag_multiplier - 1) + 1;

    for (int i = 0; i < total_number_of_columns; i++)
        parameters[i + 2] = gsl_rng_uniform_int(r, max_lag_selection - 1) + 1;

    parameters[total_number_of_columns + 2] =
        gsl_rng_uniform_int(r, max_node_selection - 1) + 1;

    clear(modelstructure);
    CreateModel(modelstructure);
    return true;
}


// ======================================================================
//  Decode chromosome → CModelStructure (single-output)
// ======================================================================

/**
 * @brief Decode internal parameter vector into a CModelStructure (single-output model).
 *
 * @details
 * Decoding process:
 *
 * 1. **Column selection**
 *    - parameters[0] is interpreted in base 2 → bitmask of selected input columns.
 *
 * 2. **Lag multiplier**
 *    - parameters[1] gives multiplier applied to lag indices.
 *
 * 3. **Lag selection**
 *    - For each input column i:
 *      parameters[i+2] decoded in base lag_frequency → lag activation mask.
 *
 * 4. **Nodes/layers selection**
 *    - Final parameter decoded in base max_number_of_nodes_in_layers.
 *
 * @param modelstructure Output pointer.
 * @return true Always true if parameters are valid.
 */
bool ModelCreator::CreateModel(CModelStructure *modelstructure) const
{
    vector<int> columns = convertToBase(parameters[0], 2);

    // Column selection
    for (unsigned int i = 0; i < columns.size(); i++)
        if (columns[i] == 1)
            modelstructure->inputcolumns.push_back(i);

    modelstructure->input_lag_multiplier = parameters[1];

    // Lag selection
    for (int i = 0; i < total_number_of_columns; i++)
    {
        vector<int> lags_onoff = convertToBase(parameters[i + 2], lag_frequency);
        vector<int> lags;

        for (unsigned int j = 0; j < lags_onoff.size(); j++)
            if (lags_onoff[j] == 1 && columns[i] == 1)
                lags.push_back(j * modelstructure->input_lag_multiplier);

        if (columns[i] == 1)
            modelstructure->lags.push_back(lags);
    }

    // Node/layer selection
    vector<int> nodes = convertToBase(parameters[2 + total_number_of_columns],
                                      max_number_of_nodes_in_layers);

    modelstructure->n_layers = nodes.size();
    modelstructure->n_nodes.resize(nodes.size());

    for (unsigned int i = 0; i < nodes.size(); i++)
        modelstructure->n_nodes[i] = nodes[i] + 1;

    return true;
}


// ======================================================================
//  Decode chromosome → CModelStructure_Multi (multi-output)
// ======================================================================

/**
 * @brief Decode parameter vector into a multi-output model structure.
 *
 * @param modelstructure Output pointer.
 * @return true Always true.
 */
bool ModelCreator::CreateModel(CModelStructure_Multi *modelstructure)
{
    modelstruct = modelstructure;
    vector<int> columns = convertToBase(parameters[0], 2);

    modelstructure->Reset();

    // Column selection
    for (unsigned int i = 0; i < columns.size(); i++)
        if (columns[i] == 1 && i < total_number_of_columns)
            modelstructure->inputcolumns.push_back(i);

    modelstructure->input_lag_multiplier = parameters[1];

    // Lag selection
    for (int i = 0; i < total_number_of_columns; i++)
    {
        vector<int> lags_onoff = convertToBase(parameters[i + 2], lag_frequency);
        vector<int> lags;

        for (unsigned int j = 0; j < lags_onoff.size(); j++)
            if (lags_onoff[j] == 1 && columns[i] == 1)
                lags.push_back(j * modelstructure->input_lag_multiplier);

        if (columns[i] == 1)
            modelstructure->lags.push_back(lags);
    }

    // Node/layer selection
    vector<int> nodes = convertToBase(parameters[2 + total_number_of_columns],
                                      max_number_of_nodes_in_layers);

    modelstructure->n_layers = nodes.size();
    modelstructure->n_nodes.resize(nodes.size());

    for (unsigned int i = 0; i < nodes.size(); i++)
        modelstructure->n_nodes[i] = nodes[i] + 1;

    return true;
}


// ======================================================================
//  SetParameters from CModelStructure
// ======================================================================

/**
 * @brief Encode a CModelStructure into an integer chromosome.
 *
 * @param modelstructure Input structure.
 * @return true if encoding succeeded.
 *
 * @details
 * Converts:
 * - Column selections → base 2 integer
 * - Lags → base lag_frequency integer
 * - Node/layer counts → base max_number_of_nodes_in_layers integer
 */
bool ModelCreator::SetParameters(CModelStructure *modelstructure)
{
    if (modelstructure->InputTimeSeries == nullptr && total_number_of_columns == 0)
    {
        cout << "Input time series or total number of columns must be provided" << endl;
        return false;
    }
    else if (total_number_of_columns != 0)
    {
        // OK: user provided total_number_of_columns
    }
    else
    {
        total_number_of_columns = modelstructure->InputTimeSeries->nvars;
    }

    parameters.resize(ParametersSize());

    // Column selection
    for (unsigned int i = 0; i < modelstructure->inputcolumns.size(); i++)
        parameters[0] += pow(2, modelstructure->inputcolumns[i]);

    parameters[1] = modelstructure->input_lag_multiplier;

    // Lag selection
    int counter = 0;
    for (int i = 0; i < total_number_of_columns; i++)
    {
        if (modelstructure->inputcolumns[counter] == i)
        {
            for (int j = 0; j < modelstructure->lags[counter].size(); j++)
            {
                parameters[i + 2] +=
                    pow(lag_frequency,
                        modelstructure->lags[counter][j] /
                        modelstructure->input_lag_multiplier);
            }
            counter++;
        }
    }

    // Node/layer encoding
    for (int i = 0; i < modelstructure->n_layers; i++)
        parameters[2 + total_number_of_columns] +=
            (modelstructure->n_nodes[i] - 1) *
            pow(max_number_of_nodes_in_layers, i);

    return true;
}


// ======================================================================
//  Parameter count
// ======================================================================

/**
 * @brief Compute the number of parameters in the chromosome representation.
 *
 * @return Integer count = 2 + Ncols + 1.
 */
int ModelCreator::ParametersSize()
{
    int out = 2;
    out += total_number_of_columns;
    out++;
    return out;
}


// ======================================================================
//  Base conversion utility
// ======================================================================

/**
 * @brief Convert integer to digit vector in a given base.
 *
 * @param number Integer to convert.
 * @param base   Base (≥ 2).
 * @return Digit vector in least-significant-digit-first order.
 */
std::vector<int> convertToBase(unsigned long int number, int base)
{
    std::vector<int> result;

    if (number == 0)
    {
        result.push_back(0);
        return result;
    }

    while (number > 0)
    {
        int remainder = number % base;
        result.push_back(remainder);
        number /= base;
    }

    return result;
}


// ======================================================================
//  CreateModel for attached FFN (no parameters passed)
// ======================================================================

/**
 * @brief Build a model using internal parameters and the FFN.ModelStructure.
 *
 * @return true Always true.
 */
bool ModelCreator::CreateModel()
{
    CreateModel(&FFN.ModelStructure);
    return true;
}


// ======================================================================
//  Fitness evaluation
// ======================================================================

/**
 * @brief Compute training and testing metrics for the current model structure.
 *
 * @return A map<string,double> containing MSE and R² for each output.
 *
 * @details
 * Steps:
 * - FFN.Initiate()
 * - FFN.Train()
 * - FFN.Test()
 * - FFN.PerformanceMetrics()
 *
 * Metrics exposed:
 * - "MSE_Train_i"
 * - "R2_Train_i"
 * - "MSE_Test_i"
 * - "R2_Test_i"
 *
 * @note Sets `initiated = true` after first use.
 */
map<string, double> ModelCreator::Fitness()
{
    map<string, double> out;

    FFN.Initiate(!initiated);
    FFN.Train();
    FFN.Test();
    FFN.PerformanceMetrics();

    for (int constituent = 0;
         constituent < modelstruct->outputcolumns.size();
         constituent++)
    {
        out["MSE_Train_" + aquiutils::numbertostring(constituent)] =
            FFN.nMSE_Train[constituent];

        out["R2_Train_" + aquiutils::numbertostring(constituent)] =
            FFN._R2_Train[constituent];

        out["MSE_Test_" + aquiutils::numbertostring(constituent)] =
            FFN.nMSE_Test[constituent];

        out["R2_Test_" + aquiutils::numbertostring(constituent)] =
            FFN._R2_Test[constituent];
    }

    initiated = true;
    return out;
}


// ======================================================================
//  Copy constructor / assignment operator
// ======================================================================

/**
 * @brief Copy constructor: deep copies FFN, params, and structural limits.
 */
ModelCreator::ModelCreator(const ModelCreator &other)
{
    FFN = other.FFN;
    initiated = other.initiated;
    total_number_of_columns = other.total_number_of_columns;
    maximum_superficial_lag = other.maximum_superficial_lag;
    lag_frequency = other.lag_frequency;
    max_number_of_nodes_in_layers = other.max_number_of_nodes_in_layers;
    max_number_of_layers = other.max_number_of_layers;
    max_lag_multiplier = other.max_lag_multiplier;
}

/**
 * @brief Assignment operator: deep copies FFN, params, and structural limits.
 *
 * @return Reference to *this.
 */
ModelCreator &ModelCreator::operator=(const ModelCreator &other)
{
    FFN = other.FFN;
    initiated = other.initiated;
    total_number_of_columns = other.total_number_of_columns;
    maximum_superficial_lag = other.maximum_superficial_lag;
    lag_frequency = other.lag_frequency;
    max_number_of_nodes_in_layers = other.max_number_of_nodes_in_layers;
    max_number_of_layers = other.max_number_of_layers;
    max_lag_multiplier = other.max_lag_multiplier;
    return *this;
}
