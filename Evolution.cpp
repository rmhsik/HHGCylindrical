#include <iostream>
#include <cmath>
#include <complex>
#include <armadillo>
#include <typeinfo>
#include <omp.h>
#include "tdma.h"
#include "tridot.h"
#include "int_simpson.h"
#define ARMA_NO_DEBUG

void CoulombPotential(arma::mat &V, arma::vec &r, arma::vec &z){

    for(int i=0;i<V.n_rows;i++){
        for(int j=0;j<V.n_cols;j++){
            V(i,j) =-1/sqrt(r(i)*r(i)+z(j)*z(j));
        }
    }
}

double envelope(double tmax, double t){
    if (t<tmax){
        return pow(sin(M_PI*t/tmax),2);
    }
    else {
        return 0.0;
    }
}

double EField(double t){
    double E0=0.067;
    double w = 0.057;
    double tmax = 4*2*M_PI/w;
    return E0*envelope(tmax,t)*sin(w*t);
}

double BField(double t){
    double B0=0.0;
    double w = 0.057;
    double tmax = 4*2*M_PI/w;
    return B0*envelope(tmax,t)*sin(w*t);
}

void Gaussian(arma::cx_mat &Psi, arma::vec &r, arma::vec &z, const double r0, const double z0, const double a ){
    for(int i=0;i<Psi.n_rows;i++){
        for(int j=0;j<Psi.n_cols;j++){
            Psi(i,j) = exp(-pow(r(i)-r0,2)/a-pow(z(j)-z0,2)/a);
        }
    }
}

void HamR(arma::cx_mat &Hr, arma::dmat &Potential, double BField, arma::vec &r, const double dr,arma::dmat &R, const int i){
    arma::cx_colvec d = 1/pow(dr,2)*arma::ones<arma::cx_colvec>(r.n_elem) + 0.5*(Potential.col(i)+1.0/8.0*pow(BField,2)*R.col(i)%R.col(i));
    arma::cx_colvec u = -1.0/(2.0*dr)*(1.0/dr*arma::ones<arma::cx_colvec>(r.n_elem)+1.0/(2.0*r));
    arma::cx_colvec l = -1.0/(2.0*dr)*(1.0/dr*arma::ones<arma::cx_colvec>(r.n_elem)-1.0/(2.0*r));

    Hr.col(0) = u;
    Hr.col(1) = d;
    Hr.col(2) = l;

    Hr.col(0)(0) = -1.0/(dr*dr);
    Hr.col(1)(0) = 1.0/(dr*dr)+0.5*Potential.col(i)(0);
    Hr.col(2)(0) = 0.0;

    Hr.col(0)(r.n_elem-1) = 0.0;
    Hr.col(2)(0) = 0.0;
}

void HamZ(arma::cx_mat &Hz, arma::dmat &Potential,double VecPot, double BField, arma::vec &z, const double dz,arma::dmat &R, const int i){
    arma::cx_colvec d = 1.0/pow(dz,2)*arma::ones<arma::cx_colvec>(z.n_elem) + 0.5*(Potential.row(i).t()+1.0/8.0*pow(BField,2)*R.row(i).t()%R.row(i).t())+0.5/(pow(137.04,2))*pow(VecPot,2);
    arma::cx_colvec u = -1.0/(2.0*dz*dz)*arma::ones<arma::cx_colvec>(z.n_elem)+std::complex<double>(0.0,1.0)/(2.0*137.04*dz)*VecPot;
    arma::cx_colvec l = -1.0/(2.0*dz*dz)*arma::ones<arma::cx_colvec>(z.n_elem)-std::complex<double>(0.0,1.0)/(2.0*137.04*dz)*VecPot;

    Hz.col(0) = u;
    Hz.col(1) = d;
    Hz.col(2) = l;

    Hz.col(0)(z.n_elem-1) = 0.0;
    Hz.col(2)(0) = 0.0;
}

void StepR(arma::cx_mat &Psi,arma::cx_mat &PsiOut, arma::dmat &V,double VecPot, double BField,arma::vec &r, const double dr, int Nr, arma::dmat &R,
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
    int j;

    //#pragma omp parallel for private(j)
    for (j=0;j<Nz;j++){
        HamR(Hr, V, BField, r, dr, R, j);
        PsiCol = Psi.col(j);
        M = Iden-std::complex<double>(0.0,1.0)/2.0*Hr*dt;
        Mp = Iden+std::complex<double>(0.0,1.0)/2.0*Hr*dt;
        //M.save("M.dat",arma::raw_ascii);
        tridot(Mp,PsiCol,b,Nr);
        tdmaSolver(M,b,PsiColNew,Nr);
        PsiNew.col(j) = PsiColNew;
    }
    PsiOut = PsiNew;
}

void StepZ(arma::cx_mat &Psi,arma::cx_mat &PsiOut, arma::dmat &V, double VecPot, double BField,arma::vec &r, const double dr, int Nr, arma::dmat &R,
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
    int j;
    //#pragma omp parallel for private(j)
   
    for (j=0;j<Nr;j++){
        HamZ(Hz, V,VecPot,BField, z, dz, R, j);
        PsiCol = Psi.row(j).t();
        M = Iden+std::complex<double>(0.0,1.0)/2.0*Hz*dt;
        Mp = Iden-std::complex<double>(0.0,1.0)/2.0*Hz*dt;
        tridot(Mp,PsiCol,b,Nz);
        tdmaSolver(M,b,PsiColNew,Nz);
        PsiNew.row(j) = PsiColNew.t();
    }
     //M.save("M.dat",arma::raw_ascii);
    PsiOut = PsiNew;
}

std::complex<double> Energy(arma::cx_mat &Psi, arma::dmat &V,double VecPot,double BField, arma::dmat &R, arma::vec &r, arma::vec &z){
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
        HamR(Hr,V, BField,r,dr,R,j);
        PsiColR = Psi.col(j);
        tridot(Hr,PsiColR,PsiOutColR,r.n_elem);
        PsiNewR.col(j) = PsiOutColR;
    }

    for (int j=0; j<r.n_elem;j++){
        HamZ(Hz,V,VecPot,BField,z,dz, R,j);
        PsiColZ = Psi.row(j).t();
        tridot(Hz,PsiColZ,PsiOutColZ,z.n_elem);
        PsiNewZ.row(j) = PsiOutColZ.t();
    }

    std::complex<double> E = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%(PsiNewR+PsiNewZ)*dr,0)*dz,1));
    return E;
}

void derivativeZ(arma::dmat &U, arma::dmat z, arma::dmat &DU){
    int Nz = z.n_elem;
    double dz = (z(z.n_elem-1)-z(0))/(double)Nz;
    
    for(int i=0;i<U.n_rows;i++){
        DU(i,0) = (U(i,1)-U(i,0))/dz;
    }
    for(int i=0;i<U.n_rows;i++){
        DU(i,U.n_cols-1) = (U(i,U.n_cols-1)-U(i,U.n_cols-2))/dz;
    }
        //std::cout<<"lol"<<std::endl;

    for(int i=0;i<U.n_rows;i++){
        for(int j=1; j<U.n_cols-1;j++){
            DU(i,j) = (U(i,j+1)-U(i,j-1))/(2.0*dz);
        }
    }
}


std::complex<double> AcceZ(arma::cx_mat &Psi, arma::dmat &V,double VecPot,double BField, arma::dmat &R, arma::vec &r, arma::vec &z){
    int Nr = r.n_elem;
    int Nz = z.n_elem;
    double dr = abs(r(Nr-1)-r(0))/(double)Nr;
    double dz = abs(z(Nz-1)-z(0))/(double)Nz;
    arma::dmat dV(Nr,Nz,arma::fill::zeros);
    derivativeZ(V,z,dV);
    std::complex<double> acc = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%(-1*dV)%(Psi)*dr,0)*dz,1));
    std::cout<<acc<<std::endl;
    return acc;
}

void maskZ(arma::cx_mat &Mask,arma::vec &r, arma::vec &z, double zb, double gamma){
    int Nz = z.n_elem;
    int Nr = r.n_elem;
    arma::cx_colvec maskvec(Nz,arma::fill::ones);

    for(int i=0;i<Nz;i++){
        if(z(i)<(z(0)+zb)){
            maskvec(i) = pow(cos(M_PI*(z(i)-(z(0)+zb))*gamma/(2*zb)),1.0/8.0);

        }

        if(z(i)>(z(Nz-1)-zb)){
            maskvec(i) = pow(cos(M_PI*(z(i)-(z(Nz-1)-zb))*gamma/(2*zb)),1.0/8.0);
        }
    }
    std::cout<<"lol \n";
    for(int i = 0; i<Nz;i++){
        Mask.col(i) = maskvec(i)*arma::ones<arma::colvec>(Nr);
    }
}

void maskR(arma::cx_mat &Mask, arma::vec &r, arma::vec &z, double rb, double gamma){
    int Nr = r.n_elem;
    int Nz = z.n_elem;
    arma::cx_rowvec maskvec(Nr,arma::fill::ones);

    for(int i=0;i<Nr;i++){        
        if(r(i)>(r(Nr-1)-rb)){
            maskvec(i) = pow(cos(M_PI*(r(i)-(r(Nr-1)-rb))*gamma/(2*rb)),1.0/4.0);
        }
    }

    for(int i = 0; i<Nr;i++){
        Mask.row(i) = maskvec(i)*arma::ones<arma::rowvec>(Nz);
    }
}


int main(){
    //omp_set_num_threads(4);

    double rmin = 0;
    double rmax = 100;
    int Nr = 1000;
    double dr = (rmax-rmin)/Nr;
    double zmin = -120;
    double zmax = 120;
    int Nz = 2400;
    double dz = (zmax-zmin)/Nz;

    double w = 0.057;
    double t0 = 0.0;
    double tmax = 4.0*2.0*M_PI/w;
    double dt = 0.02;
    int Nt = (tmax-t0)/dt;

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
    std::cout<<"tlim: "<<tmax<<std::endl;
    std::cout<<"Nt: "<<Nt<<std::endl;
    std::cout<<"dt: "<<dt<<std::endl;

    std::complex<double> Norm;

    arma::dmat t = arma::linspace(t0,tmax,Nt);
    arma::dmat ElectricField = arma::colvec(Nt,arma::fill::zeros);
    arma::dmat MagneticField = arma::colvec(Nt,arma::fill::zeros);
    arma::dmat VecPotential = arma::colvec(Nt,arma::fill::zeros);
    arma::vec r = arma::linspace(rmin,rmax,Nr);
    arma::vec z = arma::linspace(zmin,zmax,Nz);
    arma::mat V(Nr,Nz,arma::fill::zeros);
    arma::mat dV(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Psi(Nr,Nz,arma::fill::zeros);
    arma::cx_mat PsiOld(Nr,Nz,arma::fill::zeros);
    arma::cx_mat PsiR(Nr,Nz,arma::fill::zeros);
    arma::dmat Psi2(Nr,Nz,arma::fill::zeros);
    arma::cx_mat PsiZ(Nr,Nz,arma::fill::zeros);
    arma::cx_mat MaskZ(Nr,Nz,arma::fill::zeros);
    arma::cx_mat MaskR(Nr,Nz,arma::fill::zeros);
    arma::dmat R(Nr,Nz,arma::fill::zeros);
    arma::cx_colvec acc(Nt,arma::fill::zeros);
    arma::cx_colvec normVec(Nt,arma::fill::zeros);
    arma::cx_colvec enerVec(Nt,arma::fill::zeros);
    //arma::cx_mat Hr(Nr,3,arma::fill::zeros);
    //arma::cx_mat Hz(Nz,3,arma::fill::zeros);

    r = r+dr/2.0;
    for(int i = 0; i<Nr;i++){
        R.row(i) = r(i)*arma::ones<arma::rowvec>(Nz);
    }
    R.save("R.dat",arma::raw_ascii);
    //r.save("r.dat",arma::raw_ascii);
    //z.save("z.dat",arma::raw_ascii);


    for(int i=0; i<Nt;i++){
        MagneticField(i) = BField(t(i));
        ElectricField(i) = EField(t(i));
    }

    for(int i=0; i<Nt;i++){
        VecPotential(i) = -137.04*intSimpson(EField,0,t(i),6000);
    }



    CoulombPotential(V,r,z);
    //Gaussian(Psi,r,z,0.0,0.0,4.0);
    //std::complex<double> Norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1));
    //std::cout<< Norm <<std::endl;
    //Psi = Psi/sqrt(Norm);
    Psi.load("PsiGround_120_120.dat",arma::raw_ascii);
    //std::cout<< 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1))<<std::endl;

    V.save("Coulomb.dat",arma::raw_ascii);
    Psi2  = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    //Psi2.save("PsiProb.dat",arma::raw_ascii);

    //PsiOld = Psi;

    derivativeZ(V,z,dV);
    dV.save("dV.dat",arma::raw_ascii);

    std::cout<<"Mask\n";
    maskZ(MaskZ,r,z,12.0,1.0);
    std::cout<<"MaskZ\n";
    maskR(MaskR,r,z,10.0,1.0);
    std::cout<<"MaskR\n";
    MaskZ.save("MaskZ.dat",arma::raw_ascii);
    MaskR.save("MaskR.dat",arma::raw_ascii);
    std::cout<<Energy(Psi,V,VecPotential(0),MagneticField(0),R,r,z)<<std::endl;
    Norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1));
    std::cout<<"Norm: "<<Norm<<" Energy: "<<Energy(Psi,V,0.0,0.0,R,r,z)<<std::endl;

    for(int i=0;i<Nt;i++){
        StepZ(Psi,PsiZ,V,VecPotential(i),MagneticField(i),r,dr,Nr, R, z,dz,Nz,t(i),dt/2.0);
        //PsiZ = Psi%MaskZ%MaskR;
        StepR(PsiZ,PsiR,V,VecPotential(i),MagneticField(i),r,dr,Nr, R,z,dz,Nz,t(i),dt);
        //PsiR = PsiR%MaskZ%MaskR;
        StepZ(PsiR,Psi,V,VecPotential(i),MagneticField(i),r,dr,Nr, R, z,dz,Nz,t(i),dt/2.0);
        Psi = Psi%MaskZ%MaskR;

        Norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1));
        acc(i) = AcceZ(Psi,V,VecPotential(i),MagneticField(i),R,r,z);
		normVec(i) = Norm;
		enerVec(i) = Energy(Psi,V,VecPotential(i),MagneticField(i),R,r,z);
        //PsiOld = PsiR2/sqrt(Norm);
        std::cout<<"Step: "<<i<<" Norm: "<<Norm<<" Mag: "<<MagneticField(i)<<""<<" Acc: "<<acc(i)<<" Energy: "<<Energy(Psi,V,VecPotential(i),MagneticField(i),R,r,z)<<std::endl;
        //std::cout<<i<<" of "<< Nt <<std::endl;
    }
    std::cout<<"End:\n\tNorm: "<<Norm<<" Energy: "<<Energy(Psi,V,VecPotential(Nt-1),MagneticField(Nt-1),R,r,z)<<std::endl;
    Psi2 = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    Psi2.save("PsiEnd0.dat",arma::raw_ascii);
    acc.save("acc0.dat",arma::raw_ascii);
	normVec.save("normVec0.dat",arma::raw_ascii);
	enerVec.save("enerVer0.dat",arma::raw_ascii);
    MagneticField.save("MagneticField0.dat",arma::raw_ascii);
    VecPotential.save("VecPotential0.dat",arma::raw_ascii);
    ElectricField.save("ElectricField0.dat",arma::raw_ascii);
    //PsiOld.save("PsiGround.dat",arma::raw_ascii);
    //Hr.save("Hr.dat",arma::raw_ascii);
    //Hz.save("Hz.dat",arma::raw_ascii);

    return 0;
}
