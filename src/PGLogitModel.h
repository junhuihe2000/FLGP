#ifndef PGLOGITMODEL_H
#define PGLOGITMODEL_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

/*
using namespace Rcpp;
using namespace Eigen;
*/




class PGLogitModel
{
private:
  int m;
  int _N_sample = 0;
  Eigen::MatrixXd C;
  Eigen::VectorXd Y, N;
  Eigen::VectorXd kappa, omega, f;

  // polya-gamma sampler
  Rcpp::Environment pg = Rcpp::Environment::namespace_env("pgdraw");
  Rcpp::Function pgdraw = pg["pgdraw"];


  void _resample_f();
  void _resample_omega();
  void _resample_model();
  Eigen::VectorXd _collapsed_predict(const Eigen::MatrixXd & Cnv);


public:

  PGLogitModel(const Eigen::MatrixXd & _C, const Eigen::VectorXd &_Y);

  void resample_model(int N_sample = 100);

  Eigen::VectorXd predict(const Eigen::MatrixXd & Cnv);

};

#endif
