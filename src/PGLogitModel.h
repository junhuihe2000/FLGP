#pragma once

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

using namespace Rcpp;
using namespace Eigen;

//' @importFrom Rcpp sourceCpp
//' @importFrom BayesLogit rpg
//' @useDynLib FLAG, .registration=TRUE


class PGLogitModel;

Eigen::VectorXd f_to_pi(const Eigen::VectorXd & f);
double ilogit(double x);
Eigen::VectorXi pi_to_Y(const Eigen::VectorXd & pi);


// inverse logit link function
double ilogit(double x) {
  return 1/(1+exp(-x));
}

// Stick breaking transform from f to pi
Eigen::VectorXd f_to_pi(const Eigen::VectorXd & f) {
  Eigen::VectorXd pi = 1/(1+Eigen::exp(-f.array()));
  return pi;
}

// Predict Y based on pi
Eigen::VectorXi pi_to_Y(const Eigen::VectorXd & pi) {
  return (pi.array()>0.5).cast<int>();
}


class PGLogitModel
{
private:
  int m;
  int _N_sample = 0;
  Eigen::MatrixXd C;
  Eigen::VectorXi Y, N;
  Eigen::VectorXd kappa, omega, f;


  void _resample_f();
  void _resample_omega();
  void _resample_model() {
    _resample_f();
    _resample_omega();
  }
  Eigen::VectorXd _collapsed_predict(const Eigen::MatrixXd & Cnv);


public:

  PGLogitModel(const Eigen::MatrixXd & _C, const Eigen::VectorXi &_Y): C(_C), Y(_Y) {
    m = _Y.size();
    N = Eigen::VectorXi::Constant(m, 1);
    kappa = Y.cast<double>() - N.cast<double>()/2.0;
    omega = Eigen::VectorXd::Constant(m, 1);
    f = Eigen::VectorXd::Zero(m);
  }



  void resample_model(int N_sample=100) {
    _N_sample = N_sample;
    for(int i=0;i<N_sample;i++) {
      _resample_model();
    }
  }

  Eigen::VectorXd predict(const Eigen::MatrixXd & Cnv) {
    return _collapsed_predict(Cnv);
  }

};



void PGLogitModel::_resample_f() {
  Eigen::DiagonalMatrix<double,Eigen::Dynamic> sqrt_omega = omega.array().sqrt().matrix().asDiagonal();
  Eigen::MatrixXd B = sqrt_omega*C*sqrt_omega;
  B.diagonal().array() += 1;

  // compute posterior parameters
  // using equation 3.27 in GPML
  Eigen::MatrixXd sigma_post = C - C*sqrt_omega*B.llt().solve(sqrt_omega*C);
  Eigen::VectorXd mu_post = sigma_post*kappa;
  Eigen::MatrixXd L_post = sigma_post.llt().matrixL();

  // sample f from posterior Gaussian distributions
  const Eigen::Map<Eigen::VectorXd> rand_vec = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Rcpp::rnorm(m));
  f = mu_post + L_post*rand_vec;
}


void PGLogitModel::_resample_omega() {
  Environment BayesLogit = Environment::namespace_env("BayesLogit");
  Function rpg = BayesLogit["rpg"];

  // Update auxiliary variables according to their conditional Poyla-Gamma distributions
  for(int i=0;i<m;i++) {
    omega(i) = Rcpp::as<double>(rpg(Named("num")=1, Named("h")=N(i), Named("z")=f(i)));
  }
}


Eigen::VectorXd PGLogitModel::_collapsed_predict(const Eigen::MatrixXd & Cnv) {
  int m_new = Cnv.rows();
  Eigen::VectorXd mu_f_new;

  // using equation 3.27 in GPML
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> sqrt_omega = omega.array().sqrt().matrix().asDiagonal();
  Eigen::MatrixXd B = sqrt_omega*C*sqrt_omega;
  B.diagonal().array() += 1;
  mu_f_new = Cnv*(kappa - sqrt_omega*B.llt().solve(sqrt_omega*(C*kappa)));

  // convert f to pi
  Eigen::VectorXd pi_new(f_to_pi(mu_f_new));
  return pi_new;
}



