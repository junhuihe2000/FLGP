// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(nloptr)]]
#include <nloptrAPI.h>
// #include <nlopt.h>

#include <iostream>

#include "train.h"
#include "Utils.h"
#include "Spectrum.h"

using namespace Rcpp;
using namespace Eigen;


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

  return ReturnValue(t, obj);
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


