#include "ffnwrapper.h"

FFNWrapper::FFNWrapper():FFN<>()
{

}

FFNWrapper::FFNWrapper(const FFNWrapper &rhs):FFN<>(rhs)
{
    ModelStructure = rhs.ModelStructure;
    input_data = rhs.input_data;
    output_data = rhs.output_data;
}

FFNWrapper& FFNWrapper::operator=(const FFNWrapper& rhs)
{
    FFN<>::operator=(rhs);
    ModelStructure = rhs.ModelStructure;
    input_data = rhs.input_data;
    output_data = rhs.output_data;
    return *this;
}
FFNWrapper::~FFNWrapper()
{

}

bool Train()
{


    return true;
}
