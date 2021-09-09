#ifndef TDMA_H
#define TDMA_H
#include <armadillo>

    void tdmaSolver(arma::cx_mat &H, arma::cx_colvec &Psi, 
            arma::cx_colvec &Psiout, const int N);
#endif