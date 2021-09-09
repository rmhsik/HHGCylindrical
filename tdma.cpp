#include <cmath>
#include <complex>
#include <cstring>
#include "tdma.h"

 void tdmaSolver(arma::cx_mat &H, arma::cx_colvec &Psi, 
            arma::cx_colvec &Psiout, const int N){

    std::complex<double> wc;
    arma::cx_colvec ac = H.col(2);
    arma::cx_colvec bc = H.col(1);
    arma::cx_colvec cc = H.col(0);
    arma::cx_colvec dc = Psi;

    for(int i = 1; i<=N-1;i++){
        wc = ac(i)/bc(i-1);
        bc(i) = bc(i) - wc*cc(i-1);
        dc(i) = dc(i) - wc*dc(i-1);
    }

    Psiout(N-1) = dc(N-1)/bc(N-1);
    
    for(int i = N-2;i>=0;i--){
        Psiout(i) = (dc(i)-cc(i)*Psiout(i+1))/bc(i);
    }
}