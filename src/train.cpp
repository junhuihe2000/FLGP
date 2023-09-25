// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(nloptr)]]
#include <nloptrAPI.h>
// #include <nlopt.h>

#include <iostream>
#include <vector>

#include "train.h"
#include "Utils.h"
#include "Spectrum.h"

/*
using namespace Rcpp;
using namespace Eigen;
*/


double negative_log_posterior_logit_cpp(unsigned n, const double *x, double *grad, void *data) {
  PostOFData * _data = (PostOFData *) data;
  // marginal likelihood
  Eigen::MatrixXd C = HK_from_spectrum_cpp(_data->eigenpair, _data->K, x[0], _data->idx, _data->idx);
  C.diagonal().array() += _data->sigma;
  double mll = marginal_log_likelihood_logit_la_cpp(C, _data->Y, _data->N);

  // prior
  double pr = _data->p*std::log(x[0]+1e-5) + std::pow(x[0]/_data->tau,-_data->q);

  return (-mll+pr);
}


double negative_marginal_likelihood_logit_cpp(unsigned n, const double *x, double *grad, void * data) {
  MargOFData * _data = (MargOFData *) data;
  Eigen::MatrixXd C = HK_from_spectrum_cpp(_data->eigenpair, _data->K, x[0], _data->idx, _data->idx);
  C.diagonal().array() += _data->sigma;
  double mll = marginal_log_likelihood_logit_la_cpp(C, _data->Y, _data->N);
  return -mll;
}



ReturnValue train_lae_logit_gp_cpp(void *data, std::string approach,
                                   double t0, double lb, double ub) {
  // initialize t
  if(t0<0) {
    t0 = 10;
  }

  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_COBYLA, 1);
  nlopt_set_lower_bounds(opt, &lb);
  nlopt_set_upper_bounds(opt, &ub);

  // empirical Bayes
  if(approach=="marginal") {
    nlopt_set_min_objective(opt, negative_marginal_likelihood_logit_cpp, data);
  } else if(approach=="posterior") {
    nlopt_set_min_objective(opt, negative_log_posterior_logit_cpp, data);
  } else {
    Rcpp::stop("This model selection approach is not supported!");
  }

  nlopt_set_xtol_rel(opt, 1e-4);

  double t = t0;
  double obj;
  if(nlopt_optimize(opt, &t, &obj)<0) {
    std::cout << "nlopt failed!" << std::endl;
  }

  nlopt_destroy(opt);

  // negative objective function
  return ReturnValue(t, -obj);
}



double negative_log_posterior_regression_cpp(unsigned n, const double *x, double *grad, void *data) {
  PostOFDataReg * _data = (PostOFDataReg *) data;
  // marginal likelihood
  Eigen::MatrixXd C = HK_from_spectrum_cpp(_data->eigenpair, _data->K, x[0], _data->idx, _data->idx);
  C.diagonal().array() += _data->sigma;
  C.diagonal().array() += x[1];
  double mll = marginal_log_likelihood_regression_cpp(C, _data->Y);

  // prior
  double pr0 = _data->p*std::log(x[0]+1e-5) + std::pow(x[0]/_data->tau,-_data->q);
  double pr1 = (_data->alpha+1)*std::log(x[1]+1e-5) + _data->beta/x[1];

  return (-mll+pr0+pr1);
}


double negative_marginal_likelihood_regression_cpp(unsigned n, const double *x, double *grad, void * data) {
  MargOFDataReg * _data = (MargOFDataReg *) data;
  Eigen::MatrixXd C = HK_from_spectrum_cpp(_data->eigenpair, _data->K, x[0], _data->idx, _data->idx);
  C.diagonal().array() += _data->sigma;
  C.diagonal().array() += x[1];
  double mll = marginal_log_likelihood_regression_cpp(C, _data->Y);
  return -mll;
}


ReturnValueReg train_regression_gp_cpp(void *data, std::string approach,
                                       std::vector<double>* x0,
                                       std::vector<double>* lb, std::vector<double>* ub) {
  // initialize x, lower bound and upper bound
  bool new_x = false;
  bool new_lb = false;
  bool new_ub = false;
  if(x0->empty()) {
    x0 = new std::vector<double>;
    new_x = true;
    x0->push_back(10);
    x0->push_back(1e-1);
  }

  if(lb->empty()) {
    lb = new std::vector<double>;
    new_lb = true;
    lb->push_back(1e-3);
    lb->push_back(1e-3);
  }

  if(ub->empty()) {
    ub = new std::vector<double>;
    new_ub = true;
    ub->push_back(std::numeric_limits<double>::infinity());
    ub->push_back(std::numeric_limits<double>::infinity());
  }

  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_COBYLA, 2);
  nlopt_set_lower_bounds(opt, &((*lb)[0]));
  nlopt_set_upper_bounds(opt, &((*ub)[0]));

  // empirical Bayes
  if(approach=="marginal") {
    nlopt_set_min_objective(opt, negative_marginal_likelihood_regression_cpp, data);
  } else if(approach=="posterior") {
    nlopt_set_min_objective(opt, negative_log_posterior_regression_cpp, data);
  } else {
    Rcpp::stop("This model selection approach is not supported!");
  }

  nlopt_set_xtol_rel(opt, 1e-4);

  std::vector<double> x = *x0;
  double obj;
  if(nlopt_optimize(opt, &(x[0]), &obj)<0) {
    std::cout << "nlopt failed!" << std::endl;
  }

  nlopt_destroy(opt);

  if(new_x) {delete x0;}
  if(new_lb) {delete lb;}
  if(new_ub) {delete ub;}

  // negative objective function
  return ReturnValueReg(x, -obj);
}

// C = K + sigma^2*I
double marginal_log_likelihood_regression_cpp(const Eigen::MatrixXd & C,
                                              const Eigen::VectorXd & Y) {
  // Algotithm 2.1 in GPML
  Eigen::LLT<Eigen::MatrixXd> chol_C(C);
  Eigen::VectorXd alpha = chol_C.solve(Y);

  // marginal log likelihood
  double mll = 0;
  mll += -0.5*(Y.array()*alpha.array()).sum();
  mll += -Eigen::MatrixXd(chol_C.matrixL()).diagonal().array().log().sum();
  return mll;
}


double marginal_log_likelihood_logit_la_cpp(const Eigen::MatrixXd & C,
                                            const Eigen::VectorXd & Y,
                                            const Eigen::VectorXd & N,
                                            double tol,
                                            int max_iter) {
  int m = Y.rows();

  // initialize f
  Eigen::VectorXd f = Eigen::VectorXd::Constant(m, 0);

  Eigen::VectorXd pi, W, b, a, f_new;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> sqrt_W;
  Eigen::MatrixXd B;
  Eigen::LLT<Eigen::MatrixXd> chol_B;


  // Newton method
  // Algorithm 3.1 in GPML
  for(int iter=0;iter<max_iter;iter++) {
    pi = f_to_pi(f);
    W = N.array()*pi.array()*(1-pi.array());
    sqrt_W = W.array().sqrt().matrix().asDiagonal();
    B = sqrt_W*C*sqrt_W;
    B.diagonal().array() += 1;
    chol_B = B.llt();
    b = W.array()*f.array() + Y.array()*(1-pi.array()) + (N-Y).array()*(-pi.array());
    a = b - sqrt_W*chol_B.solve(sqrt_W*(C*b));
    f_new = C*a;

    if((f-f_new).lpNorm<1>()<tol) {
      f = f_new;
      break;
    } else {
      f = f_new;
    }
  }

  pi = f_to_pi(f);
  // approximate marginal log likelihood
  double amll = -0.5*(a.array()*f.array()).sum();
  amll += (Y.array()*pi.array().log()).sum() + ((N-Y).array()*(1-pi.array()).log()).sum();
  amll -= Eigen::MatrixXd(chol_B.matrixL()).diagonal().array().log().sum();

  return amll;
}


