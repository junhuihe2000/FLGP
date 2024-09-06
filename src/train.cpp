// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(nloptr)]]
#include <nloptrAPI.h>

#include <vector>

#include "train.h"
#include "Utils.h"
#include "Spectrum.h"



double negative_log_posterior_logit_cpp(unsigned n, const double *x, double *grad, void *data) {
  PostOFData * _data = (PostOFData *) data;
  // marginal likelihood
  Eigen::MatrixXd C = HK_from_spectrum_cpp(_data->eigenpair, _data->K, x[0], _data->idx, _data->idx);
  C.diagonal().array() += _data->sigma;
  double mll = marginal_log_likelihood_logit_la_cpp(C, _data->Y, _data->N);

  // prior
  double pr = _data->p*std::log(x[0]+1e-9) + std::pow(x[0]/_data->tau,-_data->q);

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
    Rcpp::Rcout << "nlopt failed!" << "\n";
  }

  nlopt_destroy(opt);

  // negative objective function
  return ReturnValue(t, -obj);
}


// ##########################################################################
// Gaussian process regression
// ##########################################################################

int count = 0;

double npll_rbf_regression_cpp(unsigned n, const double *x, double *grad, void *data) {
  PostDataRBF* _data = (PostDataRBF* ) data;

  // negative_marginal log likelihood
  double nmll = nmll_rbf_regression_cpp(n, x, grad, data);

  // prior
  double pr = (_data->alpha+1)*std::log(x[1]+_data->sigma) + _data->beta/(x[1]+_data->sigma);

  grad[1] += (_data->alpha+1)/(x[1]+_data->sigma) - _data->beta/std::pow(x[1]+_data->sigma,2);

  return (nmll+pr);
}

// Subset of Regressors
double nmll_rbf_regression_cpp(unsigned n, const double *x, double *grad, void *data) {
  ++count;
  MargDataRBF* _data = (MargDataRBF*) data;
  int m = _data->Y.rows(); int q = _data->Y.cols();

  // negative marginal log likelihood
  double nmll = 0.0;

  // using Equation 8.17 in GPML
  // RBF kernel
  Eigen::MatrixXd C_ss = Eigen::exp(-_data->dist_UU.array()/(2*x[0]));
  Eigen::MatrixXd C_ms = Eigen::exp(-_data->dist_XU.array()/(2*x[0]));
  Eigen::LLT<Eigen::MatrixXd> chol_C_ss(C_ss);

  Eigen::MatrixXd Q = (x[1]+_data->sigma)*C_ss + C_ms.transpose()*C_ms;
  Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
  Eigen::MatrixXd alpha = 1.0/(x[1]+_data->sigma)*(_data->Y - C_ms*chol_Q.solve(C_ms.transpose()*_data->Y));

  if(grad) {
    Eigen::MatrixXd grad_C_ss = C_ss.array()*(_data->dist_UU.array()/(2*x[0]*x[0]));
    Eigen::MatrixXd grad_C_ms = C_ms.array()*(_data->dist_XU.array()/(2*x[0]*x[0]));

    Eigen::MatrixXd Q_inv = chol_Q.solve(Eigen::MatrixXd::Identity(_data->s, _data->s));
    Eigen::MatrixXd C_ss_inv = chol_C_ss.solve(Eigen::MatrixXd::Identity(_data->s,_data->s));
    Eigen::MatrixXd CinvC = C_ss_inv*C_ms.transpose();
    Eigen::MatrixXd beta = CinvC*alpha;

    grad[0] = -(alpha.array()*(grad_C_ms*beta).array()).sum()/q + 0.5*(beta.array()*(grad_C_ss*beta).array()).sum()/q;
    grad[0] += 1.0/(x[1]+_data->sigma)*(CinvC.array()*grad_C_ms.transpose().array()).sum();
    grad[0] += -1.0/(x[1]+_data->sigma)*((Q_inv*C_ms.transpose()*grad_C_ms).transpose().array()*(CinvC*C_ms).array()).sum();
    grad[0] += -0.5/(x[1]+_data->sigma)*(CinvC.array()*(grad_C_ss*CinvC).array()).sum();
    grad[0] += 0.5/(x[1]+_data->sigma)*((Q_inv*C_ms.transpose()*CinvC.transpose()).transpose().array()*(grad_C_ss*CinvC*C_ms).array()).sum();

    grad[1] = -0.5*(alpha.array()*alpha.array()).sum()/q;
    grad[1] += 0.5/(x[1]+_data->sigma)*(m-(C_ms.array()*(Q_inv*C_ms.transpose()).transpose().array()).sum());
  }

  nmll += 0.5*(_data->Y.array()*alpha.array()).sum()/q;
  nmll += Eigen::MatrixXd(chol_Q.matrixL()).diagonal().array().log().sum() - Eigen::MatrixXd(chol_C_ss.matrixL()).diagonal().array().log().sum();
  nmll += 0.5*(m-_data->s)*std::log(x[1]+_data->sigma);

  return nmll;
}

double npll_rbf_diff_noise_regression_cpp(unsigned n, const double *x, double *grad, void *data) {
  PostDataRBF* _data = (PostDataRBF* ) data;
  int m = _data->Y.rows();

  // negative_marginal log likelihood
  double nmll = nmll_rbf_diff_noise_regression_cpp(n, x, grad, data);

  // prior
  double pr = 0.0;
  for(int i=1;i<=m;i++) {
    pr += ((_data->alpha+1)*std::log(x[i]+_data->sigma) + _data->beta/(x[i]+_data->sigma))/m;
    grad[i] += ((_data->alpha+1)/(x[i]+_data->sigma) - _data->beta/std::pow(x[i]+_data->sigma,2))/m;
  }

  return (nmll+pr);
}

double nmll_rbf_diff_noise_regression_cpp(unsigned n, const double *x, double *grad, void *data) {
  ++count;
  MargDataRBF* _data = (MargDataRBF*) data;
  int m = _data->Y.rows(); int q = _data->Y.cols();

  // negative marginal log likelihood
  double nmll = 0.0;

  // using Equation 8.17 in GPML
  // RBF kernel
  Eigen::MatrixXd C_ss = Eigen::exp(-_data->dist_UU.array()/(2*x[0]));
  Eigen::MatrixXd C_ms = Eigen::exp(-_data->dist_XU.array()/(2*x[0]));
  Eigen::LLT<Eigen::MatrixXd> chol_C_ss(C_ss);

  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Z(m);
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Z_inv(m);
  for(int i=1;i<=m;i++) {
    Z.diagonal()[i-1] = x[i] + _data->sigma;
    Z_inv.diagonal()[i-1] = 1.0/(x[i]+_data->sigma);
  }

  Eigen::MatrixXd Q = C_ss + C_ms.transpose()*Z_inv*C_ms;
  Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
  Eigen::MatrixXd alpha = Z_inv*_data->Y - Z_inv*C_ms*chol_Q.solve(C_ms.transpose()*Z_inv*_data->Y);

  if(grad) {
    Eigen::MatrixXd grad_C_ss = C_ss.array()*(_data->dist_UU.array()/(2*x[0]*x[0]));
    Eigen::MatrixXd grad_C_ms = C_ms.array()*(_data->dist_XU.array()/(2*x[0]*x[0]));

    Eigen::MatrixXd Q_inv = chol_Q.solve(Eigen::MatrixXd::Identity(_data->s, _data->s));
    Eigen::MatrixXd C_ss_inv = chol_C_ss.solve(Eigen::MatrixXd::Identity(_data->s,_data->s));
    Eigen::MatrixXd CinvC = C_ss_inv*C_ms.transpose();
    Eigen::MatrixXd beta = CinvC*alpha;

    grad[0] = -(alpha.array()*(grad_C_ms*beta).array()).sum()/q + 0.5*(beta.array()*(grad_C_ss*beta).array()).sum()/q;
    grad[0] += (CinvC.array()*(Z_inv*grad_C_ms).transpose().array()).sum();
    grad[0] += -((Q_inv*C_ms.transpose()*Z_inv*grad_C_ms).transpose().array()*(CinvC*Z_inv*C_ms).array()).sum();
    grad[0] += -0.5*(CinvC.array()*(grad_C_ss*CinvC*Z_inv).array()).sum();
    grad[0] += 0.5*((Q_inv*C_ms.transpose()*Z_inv*CinvC.transpose()).transpose().array()*(grad_C_ss*CinvC*Z_inv*C_ms).array()).sum();

    Eigen::MatrixXd tmp;
    for(int i=1;i<=m;i++) {
      grad[i] = -0.5*(alpha.row(i-1).array()*alpha.row(i-1).array()).sum()/q;
      tmp = C_ms.row(i-1).transpose()*Z_inv.diagonal()[i-1];
      grad[i] += 0.5*(Z_inv.diagonal()[i-1]-(tmp.array()*(Q_inv*tmp).array()).sum());
    }

  }

  nmll += 0.5*(_data->Y.array()*alpha.array()).sum()/q;
  nmll += Eigen::MatrixXd(chol_Q.matrixL()).diagonal().array().log().sum() - Eigen::MatrixXd(chol_C_ss.matrixL()).diagonal().array().log().sum();
  nmll += 0.5*(Z.diagonal().array().log().sum());

  return nmll;
}

ReturnValueReg train_rbf_regression_gp_cpp(void *data, std::string approach,
                                           std::string noise,
                                           std::vector<double>* x0,
                                           std::vector<double>* lb, std::vector<double>* ub) {
  MargDataRBF * _data = (MargDataRBF *) data;
  int m = _data->Y.rows();
  // initialize x, lower bound and upper bound
  bool new_x = false;
  bool new_lb = false;
  bool new_ub = false;
  if(noise=="same") {
    if(x0==nullptr) {
      x0 = new std::vector<double>;
      new_x = true;
      x0->push_back(1);
      x0->push_back(1);
    }

    if(lb==nullptr) {
      lb = new std::vector<double>;
      new_lb = true;
      lb->push_back(1e-4);
      lb->push_back(1e-4);
    }

    if(ub==nullptr) {
      ub = new std::vector<double>;
      new_ub = true;
      ub->push_back(std::numeric_limits<double>::infinity());
      ub->push_back(std::numeric_limits<double>::infinity());
    }
  } else if (noise=="different") {
    if(x0==nullptr) {
      x0 = new std::vector<double>;
      new_x = true;
      x0->push_back(1);
      for(int i=0;i<m;i++) {
        x0->push_back(1);
      }
    }

    if(lb==nullptr) {
      lb = new std::vector<double>;
      new_lb = true;
      lb->push_back(1e-4);
      for(int i=0;i<m;i++) {
        lb->push_back(1e-4);
      }
    }

    if(ub==nullptr) {
      ub = new std::vector<double>;
      new_ub = true;
      ub->push_back(std::numeric_limits<double>::infinity());
      for(int i=0;i<m;i++) {
        ub->push_back(std::numeric_limits<double>::infinity());
      }
    }
  } else {
    Rcpp::stop("The noise setting is illegal!");
  }

  nlopt_opt opt;
  // opt = nlopt_create(NLOPT_LN_COBYLA,2); // local derivative-free algorithm
  if(noise=="same") {
    opt = nlopt_create(NLOPT_LD_MMA, 2); // local gradient-based optimization
  } else if(noise=="different") {
    opt = nlopt_create(NLOPT_LD_MMA, m+1); // local gradient-based optimization
  }
  nlopt_set_lower_bounds(opt, &((*lb)[0]));
  nlopt_set_upper_bounds(opt, &((*ub)[0]));

  // nlopt_set_param(opt, "inner_maxeval", 10);

  // empirical Bayes
  if(noise=="same") {
    if(approach=="marginal") {
      nlopt_set_min_objective(opt, nmll_rbf_regression_cpp, data);
    } else if(approach=="posterior") {
      nlopt_set_min_objective(opt, npll_rbf_regression_cpp, data);
    } else {
      Rcpp::stop("This model selection approach is not supported!");
    }
  } else if(noise=="different") {
    if(approach=="marginal") {
      nlopt_set_min_objective(opt, nmll_rbf_diff_noise_regression_cpp, data);
    } else if(approach=="posterior") {
      nlopt_set_min_objective(opt, npll_rbf_diff_noise_regression_cpp, data);
    } else {
      Rcpp::stop("This model selection approach is not supported!");
    }
  }

  nlopt_set_xtol_rel(opt, 1e-5);

  count = 0;

  std::vector<double> x = *x0;
  double obj;
  nlopt_result res = nlopt_optimize(opt, &(x[0]), &obj);
  if(res<0) {
    Rcpp::Rcout << "nlopt failed!" << "\n";
  }
  Rcpp::Rcout << "The status is " << res << "\n";



  std::printf("found minimum after %d evaluations\n", count);

  nlopt_destroy(opt);

  if(new_x) {delete x0;}
  if(new_lb) {delete lb;}
  if(new_ub) {delete ub;}

  // negative objective function
  return ReturnValueReg(x, -obj);
}



double negative_log_posterior_regression_cpp(unsigned n, const double *x, double *grad, void *data) {
  // ++count;
  PostOFDataReg * _data = (PostOFDataReg *) data;

  // negative marginal log likelihood
  double nmll = negative_marginal_likelihood_regression_cpp(n, x, grad, _data);

  // prior
  double pr0 = _data->p*std::log(x[0]+1e-9) + std::pow(x[0]/_data->tau,-_data->q);
  double pr1 = (_data->alpha+1)*std::log(x[1]+_data->sigma) + _data->beta/(x[1]+_data->sigma);

  grad[0] += _data->p/(x[0]+1e-9) - (_data->q/_data->tau)*std::pow(x[0]/_data->tau, -_data->q-1);
  grad[1] += (_data->alpha+1)/(x[1]+_data->sigma) - _data->beta/std::pow(x[1]+_data->sigma,2);

  return (nmll+pr0+pr1);
}


double negative_marginal_likelihood_regression_cpp(unsigned n, const double *x, double *grad, void * data) {
  ++count;
  MargOFDataReg * _data = (MargOFDataReg *) data;
  int m = _data->Y.rows();
  int q = _data->Y.cols();

  // negative marginal log likelihood
  double nmll = 0.0;

  //double nmll_2 = 0.0;

  if(m<=_data->K) {
    Eigen::MatrixXd C = HK_from_spectrum_cpp(_data->eigenpair, _data->K, x[0], _data->idx, _data->idx);
    C.diagonal().array() += _data->sigma;
    C.diagonal().array() += x[1];

    Eigen::LLT<Eigen::MatrixXd> chol_C(C);
    Eigen::MatrixXd alpha = chol_C.solve(_data->Y);
    // use Equation 5.9 in GPML
    if(grad) {
      Eigen::MatrixXd C_inv = chol_C.solve(Eigen::MatrixXd::Identity(C.rows(),C.cols()));
      Eigen::MatrixXd U = alpha*alpha.transpose()/q - C_inv;
      const EigenPair & eigenpair = _data->eigenpair;
      Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(_data->K).array();
      const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
      Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(_data->K,0,_data->K-1);

      Eigen::MatrixXd grad_t = mat_indexing(eigenvectors, _data->idx, cols)*(-eigenvalues.array()*Eigen::exp(-x[0]*eigenvalues.array())).matrix().asDiagonal()\
        *mat_indexing(eigenvectors, _data->idx, cols).transpose();

      grad[0] = -0.5*(U.array()*grad_t.transpose().array()).sum();
      grad[1] = -0.5*U.trace();

      // gradient clipping
      double threshold = 10;
      if(std::abs(grad[1])>=threshold) {
        grad[1] = grad[1]/std::abs(grad[1])*threshold;
      }
    }
    // use Algorithm 2.1 in GPML
    nmll += 0.5*(_data->Y.array()*alpha.array()).sum()/q;
    nmll += (Eigen::MatrixXd(chol_C.matrixL()).diagonal().array()+1e-9).log().sum();
  } else {
    const EigenPair & eigenpair = _data->eigenpair;
    Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(_data->K).array();
    const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
    Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(_data->K,0,_data->K-1);

    Eigen::MatrixXd V = mat_indexing(eigenvectors, _data->idx, cols);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Lambda_sqrt = (Eigen::exp(-0.5*x[0]*eigenvalues.array())+0.0).matrix().asDiagonal();
    Eigen::MatrixXd VtV = V.transpose()*V;
    Eigen::MatrixXd Q = Lambda_sqrt*VtV*Lambda_sqrt;
    Q.diagonal().array() += x[1] + _data->sigma;
    Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
    Eigen::MatrixXd alpha = 1.0/(x[1]+_data->sigma)*(_data->Y - V*Lambda_sqrt*chol_Q.solve(Lambda_sqrt*(V.transpose()*_data->Y)));
    // use Equation 5.9 in GPML


    if(grad) {
      Eigen::MatrixXd Q_inv = chol_Q.solve(Eigen::MatrixXd::Identity(_data->K,_data->K));
      Eigen::DiagonalMatrix<double, Eigen::Dynamic> A = ((-eigenvalues.array()*(Eigen::exp(-x[0]*eigenvalues.array())+0.0))+0.0).matrix().asDiagonal();
      Eigen::MatrixXd Vta = V.transpose()*alpha;
      grad[0] = -0.5*(Vta.array()*(A*Vta).array()).sum()/q;

      grad[0] += 0.5/(x[1]+_data->sigma)*(A*VtV).trace();
      grad[0] += -0.5/(x[1]+_data->sigma)*((Q_inv*Lambda_sqrt*VtV).array()*(A*VtV*Lambda_sqrt).transpose().array()).sum();

      grad[1] = -0.5*(alpha.array()*alpha.array()).sum()/q;
      grad[1] += 0.5/(x[1]+_data->sigma)*(m-(Q_inv.array()*(Lambda_sqrt*VtV*Lambda_sqrt).transpose().array()).sum());

      // gradient clipping
      double threshold = 10;
      if(std::abs(grad[1])>=threshold) {
        grad[1] = grad[1]/std::abs(grad[1])*threshold;
      }
    }

    // Objective function value is wrong!
    // use Algorithm 2.1 in GPML
    nmll += 0.5*(_data->Y.array()*alpha.array()).sum()/q;
    nmll += (Eigen::MatrixXd(chol_Q.matrixL()).diagonal().array()+1e-9).log().sum();
    nmll += 0.5*(m-_data->K)*std::log(x[1]+_data->sigma);
  }

  return nmll;
}

double negative_log_posterior_diff_noise_regression_cpp(unsigned n, const double *x, double *grad, void *data) {
  // ++count;
  PostOFDataReg * _data = (PostOFDataReg *) data;
  int m = _data->Y.rows();

  // negative marginal log likelihood
  double nmll = negative_marginal_likelihood_diff_noise_regression_cpp(n, x, grad, _data);

  // prior
  double pr0 = _data->p*std::log(x[0]+1e-9) + std::pow(x[0]/_data->tau,-_data->q);
  grad[0] += _data->p/(x[0]+1e-9) - (_data->q/_data->tau)*std::pow(x[0]/_data->tau, -_data->q-1);
  double pr1 = 0.0;
  for(int i=1;i<=m;i++) {
    pr1 += ((_data->alpha+1)*std::log(x[i]+_data->sigma) + _data->beta/(x[i]+_data->sigma))/m;
    grad[i] += ((_data->alpha+1)/(x[i]+_data->sigma) - _data->beta/std::pow(x[i]+_data->sigma,2))/m;
  }


  return (nmll+pr0+pr1);
}

double negative_marginal_likelihood_diff_noise_regression_cpp(unsigned n, const double *x, double *grad, void * data) {
  ++count;
  MargOFDataReg * _data = (MargOFDataReg *) data;
  int m = _data->Y.rows();
  int q = _data->Y.cols();

  // negative marginal log likelihood
  double nmll = 0.0;

  //double nmll_2 = 0.0;

  if(m<=_data->K) {
    Eigen::MatrixXd C = HK_from_spectrum_cpp(_data->eigenpair, _data->K, x[0], _data->idx, _data->idx);
    C.diagonal().array() += _data->sigma;
    for(int i=1;i<=m;i++) {
      C.diagonal()[i-1] += x[i];
    }

    Eigen::LLT<Eigen::MatrixXd> chol_C(C);
    Eigen::MatrixXd alpha = chol_C.solve(_data->Y);
    // use Equation 5.9 in GPML
    if(grad) {
      Eigen::MatrixXd C_inv = chol_C.solve(Eigen::MatrixXd::Identity(C.rows(),C.cols()));
      Eigen::MatrixXd U = alpha*alpha.transpose()/q - C_inv;
      const EigenPair & eigenpair = _data->eigenpair;
      Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(_data->K).array();
      const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
      Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(_data->K,0,_data->K-1);

      Eigen::MatrixXd grad_t = mat_indexing(eigenvectors, _data->idx, cols)*(-eigenvalues.array()*Eigen::exp(-x[0]*eigenvalues.array())).matrix().asDiagonal()\
        *mat_indexing(eigenvectors, _data->idx, cols).transpose();

      grad[0] = -0.5*(U.array()*grad_t.transpose().array()).sum();
      for(int i=1;i<=m;i++) {
        grad[i] = -0.5*U.diagonal()[i-1];
      }

    }
    // use Algorithm 2.1 in GPML
    nmll += 0.5*(_data->Y.array()*alpha.array()).sum()/q;
    nmll += (Eigen::MatrixXd(chol_C.matrixL()).diagonal().array()+1e-9).log().sum();
  } else {
    const EigenPair & eigenpair = _data->eigenpair;
    Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(_data->K).array();
    const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
    Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(_data->K,0,_data->K-1);

    Eigen::MatrixXd V = mat_indexing(eigenvectors, _data->idx, cols);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Lambda_sqrt = (Eigen::exp(-0.5*x[0]*eigenvalues.array())+0.0).matrix().asDiagonal();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Z(m);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Z_inv(m);
    for(int i=1;i<=m;i++) {
      Z.diagonal()[i-1] = x[i]+_data->sigma;
      Z_inv.diagonal()[i-1] = 1.0/(x[i]+_data->sigma);
    }
    Eigen::MatrixXd VtZiV = V.transpose()*Z_inv*V;
    Eigen::MatrixXd Q = Lambda_sqrt*VtZiV*Lambda_sqrt;
    Q.diagonal().array() += 1.0;
    Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
    Eigen::MatrixXd alpha = Z_inv*_data->Y - Z_inv*V*Lambda_sqrt*chol_Q.solve(Lambda_sqrt*(V.transpose()*(Z_inv*_data->Y)));
    // use Equation 5.9 in GPML


    if(grad) {
      Eigen::MatrixXd Q_inv = chol_Q.solve(Eigen::MatrixXd::Identity(_data->K,_data->K));
      Eigen::DiagonalMatrix<double, Eigen::Dynamic> A = ((-eigenvalues.array()*(Eigen::exp(-x[0]*eigenvalues.array())+0.0))+0.0).matrix().asDiagonal();
      grad[0] = -0.5*(alpha.array()*((alpha.transpose()*V)*A*V.transpose()).transpose().array()).sum()/q;
      grad[0] += 0.5*(A*VtZiV).trace();
      grad[0] += -0.5*((Q_inv*Lambda_sqrt*VtZiV).array()*(A*VtZiV*Lambda_sqrt).transpose().array()).sum();

      Eigen::MatrixXd tmp;
      for(int i=1;i<=m;i++) {
        grad[i] = -0.5*(alpha.row(i-1).array()*alpha.row(i-1).array()).sum()/q;
        tmp = Z_inv.diagonal()[i-1]*V.row(i-1)*Lambda_sqrt;
        grad[i] += 0.5*(Z_inv.diagonal()[i-1]-((tmp*Q_inv).array()*tmp.array()).sum());
      }

      // gradient clipping
      double threshold = 1;
      for(int i=1;i<=m;i++) {
        if(std::abs(grad[i])>=threshold) {
          grad[i] = grad[i]/std::abs(grad[i])*threshold;
        }
      }

    }

    // Objective function value is wrong!
    // use Algorithm 2.1 in GPML
    nmll += 0.5*(_data->Y.array()*alpha.array()).sum()/q;
    nmll += (Eigen::MatrixXd(chol_Q.matrixL()).diagonal().array()+1e-9).log().sum();
    nmll += 0.5*((Z.diagonal().array()+1e-9).log().sum());
  }

  // Rcpp::Rcout << "nmll = " << nmll << "\n";
  return nmll;
}

ReturnValueReg train_regression_gp_cpp(void *data, std::string approach,
                                       std::string noise,
                                       std::vector<double>* x0,
                                       std::vector<double>* lb, std::vector<double>* ub) {
  MargOFData * _data = (MargOFData *) data;
  int m = _data->Y.rows();
  // initialize x, lower bound and upper bound
  bool new_x = false;
  bool new_lb = false;
  bool new_ub = false;
  if(noise=="same") {
    if(x0==nullptr) {
      x0 = new std::vector<double>;
      new_x = true;
      x0->push_back(10);
      x0->push_back(1);
    }

    if(lb==nullptr) {
      lb = new std::vector<double>;
      new_lb = true;
      lb->push_back(1e-3);
      lb->push_back(1e-4);
    }

    if(ub==nullptr) {
      ub = new std::vector<double>;
      new_ub = true;
      ub->push_back(std::numeric_limits<double>::infinity());
      ub->push_back(std::numeric_limits<double>::infinity());
    }
  } else if (noise=="different") {
    if(x0==nullptr) {
      x0 = new std::vector<double>;
      new_x = true;
      x0->push_back(10);
      for(int i=0;i<m;i++) {
        x0->push_back(1);
      }
    }

    if(lb==nullptr) {
      lb = new std::vector<double>;
      new_lb = true;
      lb->push_back(1e-3);
      for(int i=0;i<m;i++) {
        lb->push_back(1e-4);
      }
    }

    if(ub==nullptr) {
      ub = new std::vector<double>;
      new_ub = true;
      ub->push_back(std::numeric_limits<double>::infinity());
      for(int i=0;i<m;i++) {
        ub->push_back(std::numeric_limits<double>::infinity());
      }
    }
  } else {
    Rcpp::stop("The noise setting is illegal!");
  }

  nlopt_opt opt;
  if(noise=="same") {
    opt = nlopt_create(NLOPT_LD_MMA, 2); // local gradient-based optimization
  } else if(noise=="different") {
    opt = nlopt_create(NLOPT_LD_MMA, m+1); // local gradient-based optimization
  }
  nlopt_set_lower_bounds(opt, &((*lb)[0]));
  nlopt_set_upper_bounds(opt, &((*ub)[0]));

  // empirical Bayes
  if(noise=="same") {
    if(approach=="marginal") {
      nlopt_set_min_objective(opt, negative_marginal_likelihood_regression_cpp, data);
    } else if(approach=="posterior") {
      nlopt_set_min_objective(opt, negative_log_posterior_regression_cpp, data);
    } else {
      Rcpp::stop("This model selection approach is not supported!");
    }
  } else if(noise=="different") {
    if(approach=="marginal") {
      nlopt_set_min_objective(opt, negative_marginal_likelihood_diff_noise_regression_cpp, data);
    } else if(approach=="posterior") {
      nlopt_set_min_objective(opt, negative_log_posterior_diff_noise_regression_cpp, data);
    } else {
      Rcpp::stop("This model selection approach is not supported!");
    }
  }

  nlopt_set_xtol_rel(opt, 1e-5);

  count = 0;

  std::vector<double> x = *x0;
  double obj;
  nlopt_result res = nlopt_optimize(opt, &(x[0]), &obj);
  if(res<0) {
    Rcpp::Rcout << "nlopt failed!" << "\n";
  }
  Rcpp::Rcout << "The status is " << res << "\n";



  // std::printf("found minimum after %d evaluations\n", count);

  nlopt_destroy(opt);

  if(new_x) {delete x0;}
  if(new_lb) {delete lb;}
  if(new_ub) {delete ub;}

  // negative objective function
  return ReturnValueReg(x, -obj);
}

// C = K + noise*I
double marginal_log_likelihood_regression_cpp(const EigenPair & eigenpair,
                                              const Eigen::VectorXd & Y,
                                              const Eigen::VectorXi & idx,
                                              int K,
                                              double t,
                                              double noise,
                                              double sigma) {
  // Algorithm 2.1 and matrix inversion lemma in GPML
  int m = Y.rows();
  // marginal log likelihood
  double mll = 0.0;

  if(m<=K) {
    Eigen::MatrixXd C = HK_from_spectrum_cpp(eigenpair, K, t, idx, idx);
    C.diagonal().array() += sigma;
    C.diagonal().array() += noise;

    Eigen::LLT<Eigen::MatrixXd> chol_C(C);
    Eigen::VectorXd alpha = chol_C.solve(Y);

    mll += -0.5*(Y.array()*alpha.array()).sum();
    mll += -(Eigen::MatrixXd(chol_C.matrixL()).diagonal().array()+1e-9).log().sum();
  } else {
    Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(K).array();
    const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
    Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(K,0,K-1);

    Eigen::MatrixXd V = mat_indexing(eigenvectors, idx, cols);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Lambda_sqrt = (Eigen::exp(-0.5*t*eigenvalues.array())+1e-10).matrix().asDiagonal();
    Eigen::MatrixXd Q = Lambda_sqrt*V.transpose()*V*Lambda_sqrt;
    Q.diagonal().array() += noise + sigma;
    Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
    Eigen::VectorXd alpha = 1.0/(noise+sigma)*(Y - V*Lambda_sqrt*chol_Q.solve(Lambda_sqrt*(V.transpose()*Y)));

    mll += -0.5*(Y.array()*alpha.array()).sum();
    mll += -(Eigen::MatrixXd(chol_Q.matrixL()).diagonal().array()+1e-9).log().sum();
  }

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
  // locate posterior mode by Algorithm 3.1 in GPML
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
  amll -= (Eigen::MatrixXd(chol_B.matrixL()).diagonal().array()+1e-9).log().sum();

  return amll;
}


