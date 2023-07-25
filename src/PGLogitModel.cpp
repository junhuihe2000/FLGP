// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "PGLogitModel.h"

using namespace Rcpp;
using namespace Eigen;


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
Eigen::VectorXd pi_to_Y(const Eigen::VectorXd & pi) {
  return (pi.array()>0.5).cast<double>();
}



PGLogitModel::PGLogitModel(const Eigen::MatrixXd & _C, const Eigen::VectorXd &_Y): C(_C), Y(_Y) {
  m = _Y.size();
  N = Eigen::VectorXd::Constant(m, 1);
  kappa = Y - N/2.0;
  omega = Eigen::VectorXd::Constant(m, 1);
  f = Eigen::VectorXd::Zero(m);
}



void PGLogitModel::_resample_model() {
  _resample_f();
  _resample_omega();
}


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
  // Update auxiliary variables according to their conditional Poyla-Gamma distributions
  omega = Rcpp::as<Eigen::VectorXd>(pgdraw(Named("b")=N, Named("c")=f));
}


void PGLogitModel::resample_model(int N_sample) {
  _N_sample = N_sample;
  for(int i=0;i<N_sample;i++) {
    _resample_model();
  }
}


Eigen::VectorXd PGLogitModel::predict(const Eigen::MatrixXd & Cnv) {
  return _collapsed_predict(Cnv);
}


Eigen::VectorXd PGLogitModel::_collapsed_predict(const Eigen::MatrixXd & Cnv) {
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



