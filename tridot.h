#ifndef TRIDOT_H
#define TRIDOT_h
#include <armadillo>
    void tridot(arma::cx_mat &H, arma::cx_colvec &Psi, 
            arma::cx_colvec &Psiout, const int N);
#endif