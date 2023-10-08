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


// ##########################################################################
// Gaussian process regression
// ##########################################################################

int count = 0;

double negative_log_posterior_regression_cpp(unsigned n, const double *x, double *grad, void *data) {
  // ++count;
  PostOFDataReg * _data = (PostOFDataReg *) data;

  // negative marginal log likelihood
  double nmll = negative_marginal_likelihood_regression_cpp(n, x, grad, _data);

  // prior
  double pr0 = _data->p*std::log(x[0]+1e-5) + std::pow(x[0]/_data->tau,-_data->q);
  double pr1 = (_data->alpha+1)*std::log(x[1]+1e-5) + _data->beta/(x[1]+1e-5);

  grad[0] += _data->p/(x[0]+1e-5) - (_data->q/_data->tau)*std::pow(x[0]/_data->tau, -_data->q-1);
  grad[1] += (_data->alpha+1)/(x[1]+1e-5) - _data->beta/(x[1]*x[1]);

  return (nmll+pr0+pr1);
}


double negative_marginal_likelihood_regression_cpp(unsigned n, const double *x, double *grad, void * data) {
  ++count;
  MargOFDataReg * _data = (MargOFDataReg *) data;
  int m = _data->Y.rows();

  // negative marginal log likelihood
  double nmll = 0.0;

  //double nmll_2 = 0.0;

  if(m<=_data->K) {
    Eigen::MatrixXd C = HK_from_spectrum_cpp(_data->eigenpair, _data->K, x[0], _data->idx, _data->idx);
    C.diagonal().array() += _data->sigma;
    C.diagonal().array() += x[1];

    Eigen::LLT<Eigen::MatrixXd> chol_C(C);
    Eigen::VectorXd alpha = chol_C.solve(_data->Y);
    // use Equation 5.9 in GPML
    if(grad) {
      Eigen::MatrixXd C_inv = chol_C.solve(Eigen::MatrixXd::Identity(C.rows(),C.cols()));
      Eigen::MatrixXd U = alpha*alpha.transpose() - C_inv;
      const EigenPair & eigenpair = _data->eigenpair;
      Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(_data->K).array();
      const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
      Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(_data->K,0,_data->K-1);

      Eigen::MatrixXd grad_t = mat_indexing(eigenvectors, _data->idx, cols)*(-eigenvalues.array()*Eigen::exp(-x[0]*eigenvalues.array())).matrix().asDiagonal()\
        *mat_indexing(eigenvectors, _data->idx, cols).transpose();

      grad[0] = -0.5*(U.array()*grad_t.transpose().array()).sum();
      grad[1] = -0.5*U.trace();
    }
    // use Algorithm 2.1 in GPML
    nmll += 0.5*(_data->Y.array()*alpha.array()).sum();
    nmll += Eigen::MatrixXd(chol_C.matrixL()).diagonal().array().log().sum();
  } else {
    /*
    Eigen::VectorXd eigenvalues = 1.0 - _data->eigenpair.values.head(_data->K).array();
    const Eigen::MatrixXd & eigenvectors = _data->eigenpair.vectors;
    Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(_data->K,0,_data->K-1);

    Eigen::MatrixXd V = mat_indexing(eigenvectors, _data->idx, cols);
    Eigen::MatrixXd Q = V.transpose()*V;
    Q.diagonal().array() += (x[1]+_data->sigma)*Eigen::exp(x[0]*eigenvalues.array());
    Q.diagonal().array() += _data->sigma;
    Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
    Eigen::VectorXd alpha = 1.0/(x[1]+_data->sigma)*(_data->Y - V*chol_Q.solve(V.transpose()*_data->Y));
    */
    const EigenPair & eigenpair = _data->eigenpair;
    Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(_data->K).array();
    const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
    Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(_data->K,0,_data->K-1);

    Eigen::MatrixXd V = mat_indexing(eigenvectors, _data->idx, cols);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Lambda_sqrt = (Eigen::exp(-0.5*x[0]*eigenvalues.array())+0.0).matrix().asDiagonal();
    Eigen::MatrixXd Q = Lambda_sqrt*V.transpose()*V*Lambda_sqrt;
    Q.diagonal().array() += x[1] + _data->sigma;
    Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
    Eigen::VectorXd alpha = 1.0/(x[1]+_data->sigma)*(_data->Y - V*Lambda_sqrt*chol_Q.solve(Lambda_sqrt*(V.transpose()*_data->Y)));
    // use Equation 5.9 in GPML


    /*
    //--------------------------------------------------------------------
    Eigen::MatrixXd C = HK_from_spectrum_cpp(_data->eigenpair, _data->K, x[0], _data->idx, _data->idx);
    C.diagonal().array() += _data->sigma;
    C.diagonal().array() += x[1];

    Eigen::LLT<Eigen::MatrixXd> chol_C(C);
    Eigen::VectorXd beta = chol_C.solve(_data->Y);
    */

    if(grad) {
      Eigen::MatrixXd Q_inv = chol_Q.solve(Eigen::MatrixXd::Identity(_data->K,_data->K));
      Eigen::DiagonalMatrix<double, Eigen::Dynamic> A = ((-eigenvalues.array()*(Eigen::exp(-x[0]*eigenvalues.array())+0.0))+0.0).matrix().asDiagonal();
      grad[0] = -0.5*(alpha.array()*((alpha.transpose()*V)*A*V.transpose()).transpose().array()).sum();
      Eigen::MatrixXd VtV = V.transpose()*V;
      grad[0] += 0.5/(x[1]+_data->sigma)*(A*VtV).trace();
      grad[0] += -0.5/(x[1]+_data->sigma)*((Q_inv*Lambda_sqrt*VtV).array()*(A*VtV*Lambda_sqrt).transpose().array()).sum();

      grad[1] = -0.5*(alpha.array()*alpha.array()).sum();
      grad[1] += 0.5/(x[1]+_data->sigma)*(m-(Q_inv.array()*(Lambda_sqrt*VtV*Lambda_sqrt).transpose().array()).sum());

      //------------------------------------------------------------------------------------------
      /*

      Eigen::MatrixXd C_inv = chol_C.solve(Eigen::MatrixXd::Identity(C.rows(),C.cols()));
      Eigen::MatrixXd U = beta*beta.transpose() - C_inv;

      Eigen::MatrixXd grad_t = mat_indexing(eigenvectors, _data->idx, cols)*(-eigenvalues.array()*Eigen::exp(-x[0]*eigenvalues.array())).matrix().asDiagonal()\
        *mat_indexing(eigenvectors, _data->idx, cols).transpose();

      double grad_0 = -0.5*(U.array()*grad_t.transpose().array()).sum();
      double grad_1 = -0.5*U.trace();



      std::cout << "Matrix inversion lemma: grad_t = " << grad[0] << ", grad_sigma = " << grad[1] << std::endl;
      std::cout << "Gaussian process: grad_t = " << grad_0 << ", grad_sigma = " << grad_1 << std::endl;

      */

      /*
      Eigen::MatrixXd Q_inv = chol_Q.solve(Eigen::MatrixXd::Identity(_data->K,_data->K));
      Eigen::DiagonalMatrix<double, Eigen::Dynamic> A = (-eigenvalues.array()*Eigen::exp(-x[0]*eigenvalues.array())).matrix().asDiagonal();
      grad[0] = -0.5*(alpha.array()*((alpha.transpose()*V)*A*V.transpose()).transpose().array()).sum();
      Eigen::MatrixXd VtV = V.transpose()*V;
      grad[0] += 0.5/(x[1]+_data->sigma)*(A*VtV).trace();
      grad[0] += -0.5/(x[1]+_data->sigma)*((Q_inv*VtV).array()*(A*VtV).transpose().array()).sum();

      grad[1] = -0.5*(alpha.array()*alpha.array()).sum();
      grad[1] += 0.5/(x[1]+_data->sigma)*(m-(Q_inv.array()*VtV.transpose().array()).sum());
      */

      /*
      Eigen::MatrixXd Q_inv = chol_Q.solve(Eigen::MatrixXd::Identity(_data->K,_data->K));
      Eigen::MatrixXd C_inv = 1.0/(x[1]+_data->sigma)*(Eigen::MatrixXd::Identity(m,m)-V*Q_inv*V.transpose());
      Eigen::MatrixXd U = alpha*alpha.transpose() - C_inv;

      Eigen::MatrixXd grad_t = mat_indexing(eigenvectors, _data->idx, cols)*(-eigenvalues.array()*Eigen::exp(-x[0]*eigenvalues.array())).matrix().asDiagonal()\
        *mat_indexing(eigenvectors, _data->idx, cols).transpose();

      grad[0] = -0.5*(U.array()*grad_t.transpose().array()).sum();
      grad[1] = -0.5*U.trace();
      */
    }

    // Objective function value is wrong!
    // use Algorithm 2.1 in GPML
    nmll += 0.5*(_data->Y.array()*alpha.array()).sum();
    nmll += (Eigen::MatrixXd(chol_Q.matrixL()).diagonal().array()).log().sum();
    nmll += 0.5*(m-_data->K)*std::log(x[1]+_data->sigma);
    /*
    nmll += 0.5*(_data->Y.array()*alpha.array()).sum();
    double logdet_K = -x[0]*eigenvalues.array().sum() + 2 * Eigen::MatrixXd(chol_Q.matrixL()).diagonal().array().log().sum();
    nmll += 0.5*logdet_K;
    nmll += (m-_data->K)*std::log(x[1]+1e-5);
    */
    //nmll_2 = 0.5*(_data->Y.array()*beta.array()).sum() + (Eigen::MatrixXd(chol_C.matrixL()).diagonal().array()+0.0).log().sum();
  }


  /*
  Eigen::MatrixXd C = HK_from_spectrum_cpp(_data->eigenpair, _data->K, x[0], _data->idx, _data->idx);
  C.diagonal().array() += _data->sigma;
  C.diagonal().array() += x[1];
  */
  /*
  double mll = marginal_log_likelihood_regression_cpp(C, _data->Y);
  return -mll;
  */


  /*
  std::cout << "The " << count << " iteration: " << " t = " << x[0] << ", sigma = " \
            << x[1] << ", obj = " << nmll << std::endl;
  std::cout << "obj = " << nmll_2 << std::endl;
  */

  return nmll;
}


ReturnValueReg train_regression_gp_cpp(void *data, std::string approach,
                                       std::vector<double>* x0,
                                       std::vector<double>* lb, std::vector<double>* ub) {
  // initialize x, lower bound and upper bound
  bool new_x = false;
  bool new_lb = false;
  bool new_ub = false;
  if(x0==nullptr) {
    x0 = new std::vector<double>;
    new_x = true;
    x0->push_back(10);
    x0->push_back(1);
  }

  if(lb==nullptr) {
    lb = new std::vector<double>;
    new_lb = true;
    lb->push_back(1e-5);
    lb->push_back(1e-5);
  }

  if(ub==nullptr) {
    ub = new std::vector<double>;
    new_ub = true;
    ub->push_back(std::numeric_limits<double>::infinity());
    ub->push_back(std::numeric_limits<double>::infinity());
  }

  nlopt_opt opt;
  // opt = nlopt_create(NLOPT_LN_COBYLA, 2); // local derivative-free algorithm
  opt = nlopt_create(NLOPT_LD_MMA, 2); // local gradient-based optimization
  nlopt_set_lower_bounds(opt, &((*lb)[0]));
  nlopt_set_upper_bounds(opt, &((*ub)[0]));

  // nlopt_set_param(opt, "inner_maxeval", 10);

  // empirical Bayes
  if(approach=="marginal") {
    nlopt_set_min_objective(opt, negative_marginal_likelihood_regression_cpp, data);
  } else if(approach=="posterior") {
    nlopt_set_min_objective(opt, negative_log_posterior_regression_cpp, data);
  } else {
    Rcpp::stop("This model selection approach is not supported!");
  }

  nlopt_set_xtol_rel(opt, 1e-3);

  count = 0;

  std::vector<double> x = *x0;
  double obj;
  nlopt_result res = nlopt_optimize(opt, &(x[0]), &obj);
  if(res<0) {
    std::cout << "nlopt failed!" << std::endl;
  }
  std::cout << "The status is " << res << std::endl;



  std::printf("found minimum after %d evaluations\n", count);

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
    mll += -Eigen::MatrixXd(chol_C.matrixL()).diagonal().array().log().sum();
  } else {
    /*
    Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(K).array();
    const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
    Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(K,0,K-1);

    Eigen::MatrixXd V = mat_indexing(eigenvectors, idx, cols);
    Eigen::MatrixXd Q = V.transpose()*V;
    Q.diagonal().array() += (noise+sigma)*Eigen::exp(t*eigenvalues.array());
    // Eigen::MatrixXd Q = (noise+sigma)*Eigen::exp(t*eigenvalues.array()).matrix().asDiagonal() + V.transpose()*V;
    Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
    Eigen::VectorXd alpha = 1.0/(noise+sigma)*(Y - V*chol_Q.solve(V.transpose()*Y));

    mll += -0.5*(Y.array()*alpha.array()).sum();
    double logdet_K = -t*eigenvalues.array().sum() + 2 * Eigen::MatrixXd(chol_Q.matrixL()).diagonal().array().log().sum();
    mll += -0.5*logdet_K;
    */
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
    mll += -(Eigen::MatrixXd(chol_Q.matrixL()).diagonal().array()+1e-5).log().sum();
  }

  /*
  Eigen::LLT<Eigen::MatrixXd> chol_C(C);
  Eigen::VectorXd alpha = chol_C.solve(Y);

  // marginal log likelihood
  double mll = 0;
  mll += -0.5*(Y.array()*alpha.array()).sum();
  mll += -Eigen::MatrixXd(chol_C.matrixL()).diagonal().array().log().sum();
  */
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
  amll -= Eigen::MatrixXd(chol_B.matrixL()).diagonal().array().log().sum();

  return amll;
}


