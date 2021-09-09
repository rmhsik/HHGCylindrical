#include <iostream>
#include <cmath>
#include <complex>
#include <armadillo>
#include <typeinfo>
#include "tdma.h"
#include "tridot.h"
#define ARMA_NO_DEBUG

void CoulombPotential(arma::mat &V, arma::vec &r, arma::vec &z){

    for(int i=0;i<V.n_rows;i++){
        for(int j=0;j<V.n_cols;j++){
            V(i,j) = -1/sqrt(r(i)*r(i)+z(j)*z(j));
        }
    }
}

void Gaussian(arma::cx_mat &Psi, arma::vec &r, arma::vec &z, const double r0, const double z0, const double a ){
    for(int i=0;i<Psi.n_rows;i++){
        for(int j=0;j<Psi.n_cols;j++){
            //Psi(i,j) = exp(-pow(r(i)-r0,2)/a-pow(z(j)-z0,2)/a);
            Psi(i,j)=exp(-sqrt(pow(r(i)-r0,2)+pow(z(j)-z0,2)));
        }
    }
}

void HamR(arma::cx_mat &Hr, arma::dmat &Potential, arma::vec &r, const double dr, const int i){
    arma::cx_colvec d = 1/pow(dr,2)*arma::ones<arma::cx_colvec>(r.n_elem) + 0.5*Potential.col(i);
    arma::cx_colvec u = -1.0/(2.0*dr)*(1.0/dr*arma::ones<arma::cx_colvec>(r.n_elem)+1.0/(2.0*r));
    arma::cx_colvec l = -1.0/(2.0*dr)*(1.0/dr*arma::ones<arma::cx_colvec>(r.n_elem)-1.0/(2.0*r));

    Hr.col(0) = u;
    Hr.col(1) = d;
    Hr.col(2) = l;

    Hr.col(0)(0) = -1.0/(dr*dr);
    Hr.col(1)(0) = 1.0/(dr*dr)+0.5*Potential.col(i)(0);
    //Hr.col(2)(0) = 0;

}

void HamZ(arma::cx_mat &Hz, arma::dmat &Potential, arma::vec &z, const double dz, const int i){
    arma::cx_colvec d = 1/pow(dz,2)*arma::ones<arma::cx_colvec>(z.n_elem) + 0.5*Potential.row(i).t();
    arma::cx_colvec u = -1.0/(2.0*dz*dz)*arma::ones<arma::cx_colvec>(z.n_elem);
    arma::cx_colvec l = -1.0/(2.0*dz*dz)*arma::ones<arma::cx_colvec>(z.n_elem);

    Hz.col(0) = u;
    Hz.col(1) = d;
    Hz.col(2) = l;

    //Hz.col(0)(z.n_elem-1) = 0.0;
    //Hz.col(2)(0) = 0.0;
}

void StepR(arma::cx_mat &Psi,arma::cx_mat &PsiOut, arma::dmat &V,arma::vec &r, const double dr, int Nr,
           arma::vec &z, const double dz, int Nz, const double t, const std::complex<double> dt){

    arma::cx_mat PsiNew(Nr,Nz,arma::fill::zeros);
    arma::cx_mat M(Nr,3,arma::fill::zeros);
    arma::cx_mat Mp(Nr,3,arma::fill::zeros);
    arma::cx_mat Hr(Nr,3,arma::fill::zeros);
    arma::cx_mat Iden(Nr,3,arma::fill::zeros);
    Iden.col(1) = arma::ones<arma::cx_colvec>(Nr);
    arma::cx_colvec b(Nr);
    arma::cx_colvec PsiCol(Nr);
    arma::cx_colvec PsiColNew(Nr);


    for (int j=0;j<Nz;j++){
        HamR(Hr, V, r, dr, j);
        PsiCol = Psi.col(j);
        M = Iden+std::complex<double>(0.0,1.0)/2.0*Hr*dt;
        Mp = Iden-std::complex<double>(0.0,1.0)/2.0*Hr*dt;
        tridot(Mp,PsiCol,b,Nr);
        tdmaSolver(M,b,PsiColNew,Nr);
        PsiNew.col(j) = PsiColNew;
    }
    PsiOut = PsiNew;
}

void StepZ(arma::cx_mat &Psi,arma::cx_mat &PsiOut, arma::dmat &V,arma::vec &r, const double dr, int Nr,
           arma::vec &z, const double dz, int Nz, const double t, const std::complex<double> dt){

    arma::cx_mat PsiNew(Nr,Nz,arma::fill::zeros);
    arma::cx_mat M(Nz,3,arma::fill::zeros);
    arma::cx_mat Mp(Nz,3,arma::fill::zeros);
    arma::cx_mat Hz(Nz,3,arma::fill::zeros);
    arma::cx_mat Iden(Nz,3,arma::fill::zeros);
    Iden.col(1) = arma::ones<arma::cx_colvec>(Nz);
    arma::cx_colvec b(Nz);
    arma::cx_colvec PsiCol(Nz);
    arma::cx_colvec PsiColNew(Nz);


    for (int j=0;j<Nr;j++){
        HamZ(Hz, V, z, dz, j);
        PsiCol = Psi.row(j).t();
        M = Iden+std::complex<double>(0.0,1.0)/2.0*Hz*dt;
        Mp = Iden-std::complex<double>(0.0,1.0)/2.0*Hz*dt;
        tridot(Mp,PsiCol,b,Nz);
        tdmaSolver(M,b,PsiColNew,Nz);
        PsiNew.row(j) = PsiColNew.t();
    }
    PsiOut = PsiNew;
}

std::complex<double> Energy(arma::cx_mat &Psi, arma::dmat &V, arma::dmat &R, arma::vec &r, arma::vec &z){
    arma::cx_mat PsiNewR(r.n_elem,z.n_elem,arma::fill::zeros);
    arma::cx_mat PsiNewZ(r.n_elem,z.n_elem,arma::fill::zeros);
    arma::cx_mat Hr(r.n_elem,3,arma::fill::zeros);
    arma::cx_mat Hz(z.n_elem,3,arma::fill::zeros);
    arma::cx_colvec PsiColR(r.n_elem);
    arma::cx_colvec PsiOutColR(r.n_elem);
    arma::cx_colvec PsiColZ(z.n_elem);
    arma::cx_colvec PsiOutColZ(z.n_elem);


    double dr = (r(r.n_elem-1)-r(0))/r.n_elem;
    double dz = (z(z.n_elem-1)-z(0))/z.n_elem;
    

    for (int j=0; j<z.n_elem;j++){
        HamR(Hr,V,r,dr,j);
        PsiColR = Psi.col(j);
        tridot(Hr,PsiColR,PsiOutColR,r.n_elem);
        PsiNewR.col(j) = PsiOutColR;
    }

    for (int j=0; j<r.n_elem;j++){
        HamZ(Hz,V,z,dz,j);
        PsiColZ = Psi.row(j).t();
        tridot(Hz,PsiColZ,PsiOutColZ,z.n_elem);
        PsiNewZ.row(j) = PsiOutColZ.t();
    }

    std::complex<double> E = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%(PsiNewR+PsiNewZ)*dr,0)*dz,1));
    return E;
}


void maskZ(arma::dmat &Mask,arma::vec &r, arma::vec &z, double zb, double gamma){
    int Nz = z.n_elem;
    int Nr = r.n_elem;
    arma::colvec maskvec(Nz,arma::fill::ones);

    for(int i=0;i<Nz;i++){
        if(z(i)<(z(0)+zb)){
            maskvec(i) = pow(cos(M_PI*(z(i)-(z(0)+zb))*gamma/(2*zb)),1.0/8.0);

        }

        if(z(i)>(z(Nz-1)-zb)){
            maskvec(i) = pow(cos(M_PI*(z(i)-(z(Nz-1)-zb))*gamma/(2*zb)),1.0/8.0);
        }
    }
    for(int i = 0; i<Nz;i++){
        Mask.col(i) = maskvec(i)*arma::ones<arma::colvec>(Nr);
    }
}

void maskR(arma::dmat &Mask, arma::vec &r, arma::vec &z, double rb, double gamma){
    int Nr = r.n_elem;
    int Nz = z.n_elem;
    arma::rowvec maskvec(Nr,arma::fill::ones);

    for(int i=0;i<Nr;i++){        
        if(r(i)>(r(Nr-1)-rb)){
            maskvec(i) = pow(cos(M_PI*(r(i)-(r(Nr-1)-rb))*gamma/(2*rb)),1.0/8.0);
        }
    }

    for(int i = 0; i<Nr;i++){
        Mask.row(i) = maskvec(i)*arma::ones<arma::rowvec>(Nz);
    }
}

int main(){
    double rmin = 0;
    double rmax = 100;
    int Nr = 1000;
    double dr = (rmax-rmin)/Nr;
    double zmin = -120;
    double zmax = 120;
    int Nz = 2400;
    double dz = (zmax-zmin)/Nz;
    int Nt = 200;
    double tlim = 10.0;
    double dt = 0.001;

    std::cout<<"Parameters:\n";
    std::cout<<"-------------\n";
    std::cout<<"rmin: "<<rmin<<std::endl;
    std::cout<<"rmax: "<<rmax<<std::endl;
    std::cout<<"Nr: "<<Nr<<std::endl;
    std::cout<<"dr: "<<dr<<std::endl;
    std::cout<<"zmin: "<<zmin<<std::endl;
    std::cout<<"zmax: "<<zmax<<std::endl;
    std::cout<<"Nz: "<<Nz<<std::endl;
    std::cout<<"dz: "<<dz<<std::endl;
    std::cout<<"tlim: "<<tlim<<std::endl;
    std::cout<<"Nt: "<<Nt<<std::endl;
    std::cout<<"dt: "<<dt<<std::endl;

    arma::vec r = arma::linspace(rmin,rmax,Nr);
    arma::vec z = arma::linspace(zmin,zmax,Nz);
    arma::mat V(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Psi(Nr,Nz,arma::fill::zeros);
    arma::cx_mat PsiOld(Nr,Nz,arma::fill::zeros);
    arma::cx_mat PsiR(Nr,Nz,arma::fill::zeros);
    arma::cx_mat PsiR2(Nr,Nz,arma::fill::zeros);
    arma::cx_mat PsiZ(Nr,Nz,arma::fill::zeros);
    arma::dmat MaskZ(Nr,Nz,arma::fill::zeros);
    arma::dmat MaskR(Nr,Nz,arma::fill::zeros);
    arma::dmat R(Nr,Nz,arma::fill::zeros);
    //arma::cx_mat Hr(Nr,3,arma::fill::zeros);
    //arma::cx_mat Hz(Nz,3,arma::fill::zeros);

    r = r+dr/2.0;
    for(int i = 0; i<Nr;i++){
        R.row(i) = r(i)*arma::ones<arma::rowvec>(Nz);
    }
    R.save("R.dat",arma::raw_ascii);
    //r.save("r.dat",arma::raw_ascii);
    //z.save("z.dat",arma::raw_ascii);


    CoulombPotential(V,r,z);
    Gaussian(Psi,r,z,0.0,0.0,1.0);
    std::complex<double> Norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1));
    std::complex<double> rExpected;
    std::cout<< Norm <<std::endl;
    Psi = Psi/sqrt(Norm);
    std::cout<< 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1))<<std::endl;

    V.save("Coulomb.dat",arma::raw_ascii);
    arma::mat Psi2  = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    Psi2.save("PsiProb.dat",arma::raw_ascii);

    PsiOld = Psi;

    std::cout<<"Mask\n";
    maskZ(MaskZ,r,z,12.0,1.0);
    std::cout<<"MaskZ\n";
    maskR(MaskR,r,z,rmax*0.1,1.0);
    std::cout<<"MaskR\n";
    MaskZ.save("MaskZ.dat",arma::raw_ascii);
    MaskR.save("MaskR.dat",arma::raw_ascii);
    std::cout<<Energy(Psi,V,R,r,z)<<std::endl;
    for(int i=0;i<Nt;i++){
        StepR(PsiOld,PsiR,V,r,dr,Nr,z,dz,Nz,0.0,std::complex<double>(0.0,-1.0)*dt/2.0);
        StepZ(PsiR,PsiZ,V,r,dr,Nr,z,dz,Nz,0.0,std::complex<double>(0.0,-1.0)*dt);
        StepR(PsiZ,PsiR2,V,r,dr,Nr,z,dz,Nz,0.0,std::complex<double>(0.0,-1.0)*dt/2.0);
        PsiR2 = PsiR2%MaskZ%MaskR;
        Norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(PsiR2)%PsiR2*dr,0)*dz,1));
        PsiOld = PsiR2/sqrt(Norm);
        std::cout<<i<<": ";
        std::cout<<Energy(PsiOld,V,R,r,z)<<" "<<Norm<<std::endl;
    }
    Psi2  = arma::conv_to<arma::dmat>::from(arma::conj(PsiOld)%PsiOld);
    Psi2.save("PsiGround2_120_120.dat",arma::raw_ascii);
    PsiOld.save("PsiGround_120_120.dat",arma::raw_ascii);
    //Hr.save("Hr.dat",arma::raw_ascii);
    //Hz.save("Hz.dat",arma::raw_ascii);
    return 0;
}
