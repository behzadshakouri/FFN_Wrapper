TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += core
CONFIG += PRECOMPILED_HEADER
DEFINES += GSL

#CONFIG += Behzad
#DEFINES += Behzad

CONFIG += Arash
DEFINES += Arash

Behzad {
    OHQPATH = /home/behzad/Projects/OpenHydroQual/aquifolium

}

Arash {
    OHQPATH = /home/arash/Projects/QAquifolium/aquifolium

}

SOURCES += \
        $$OHQPATH/src/Utilities.cpp \
        cmodelstructure.cpp \
        cmodelstructure_multi.cpp \
        ffnwrapper.cpp \
        ffnwrapper_multi.cpp \
        main.cpp \
        modelcreator.cpp

DEFINES += ARMA_USE_LAPACK ARMA_USE_BLAS _ARMA




DEFINES += use_VTK ARMA_USE_SUPERLU
CONFIG += use_VTK

INCLUDEPATH += $$OHQPATH/include
INCLUDEPATH += $$OHQPATH/include/GA
INCLUDEPATH += $$OHQPATH/src
INCLUDEPATH += /usr/local/include

LIBS += -larmadillo -llapack -lblas -lgsl -lboost_filesystem -lboost_system -lboost_iostreams

HEADERS += \
    cmodelstructure.h \
    cmodelstructure_multi.h \
    ffnwrapper.h \
    ffnwrapper_multi.h \
    modelcreator.h
