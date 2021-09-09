#include <cmath>
#include <complex>
#include "tridot.h"

void tridot(arma::cx_mat &H, arma::cx_colvec &Psi, 
            arma::cx_colvec &Psiout, const int N){

        Psiout(0) = H.col(1)(0)*Psi(0) + H.col(0)(0)*Psi(1) ;
        Psiout(N-1) = H.col(2)(N-1)*Psi(N-2) + H.col(1)(N-1)*Psi(N-1);

        for(int i = 1; i<N-1; i++){
            Psiout(i) = H.col(2)(i)*Psi(i-1) + H.col(1)(i)*Psi(i) + H.col(0)(i)*Psi(i+1);
        }
}