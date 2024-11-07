TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

#DEFINES += GSL

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
        ffnwrapper.cpp \
        main.cpp \
        modelcreator.cpp

DEFINES += ARMA_USE_LAPACK ARMA_USE_BLAS _ARMA




DEFINES += use_VTK ARMA_USE_SUPERLU
CONFIG += use_VTK

INCLUDEPATH += $$OHQPATH/include
INCLUDEPATH += $$OHQPATH/include/GA
INCLUDEPATH += $$OHQPATH/src
INCLUDEPATH += /usr/local/include

LIBS += -larmadillo -llapack -lblas -lgsl

HEADERS += \
    ffnwrapper.h \
    modelcreator.h
