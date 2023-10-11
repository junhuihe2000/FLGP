#ifndef TRAIN_H
#define TRAIN_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <vector>

#include "Spectrum.h"





//' Mode-finding for binary Laplace GPC with logit link function
//'
//' @param C A numeric matrix with dim(m,m), covariance matrix.
//' @param Y A numeric vector with length(m), count of the positive class.
//' @param N A numeric vector with length(m), total count.
//' @param tol A double, convergence criterion, the defaulting value is `1e-5`.
//' @param max_iter An integer, the maximum iteration number, defaulting value `100`.
//'
//' @return `amll` A double, the Laplace approximation of the marginal log likelihood.
//' @export
//'
//' @examples
//' A <- matrix(rnorm(3*3),3,3)
//' C <- A%*%t(A)
//' Y <- sample(c(0,1), 3,replace=TRUE)
//' N <- rep(1,3)
//' marginal_log_likelihood_logit_la_cpp(C, Y, N)
// [[Rcpp::export(marginal_log_likelihood_logit_la_cpp)]]
double marginal_log_likelihood_logit_la_cpp(const Eigen::MatrixXd & C,
                                            const Eigen::VectorXd & Y,
                                            const Eigen::VectorXd & N,
                                            double tol = 1e-5,
                                            int max_iter = 100);

// Marginal log likelihood function for GPR
double marginal_log_likelihood_regression_cpp(const EigenPair & eigenpair,
                                              const Eigen::VectorXd & Y,
                                              const Eigen::VectorXi & idx,
                                              int K,
                                              double t,
                                              double noise,
                                              double sigma = 1e-3);

/*
//' Create negative log marginal likelihood functional for logistic regression
//'
//' @param eigenpair A list includes values and vectors.
//' @param Y A numeric vector with length(m), count of the positive class.
//' @param idx An integer vector with length(m), the index of training samples.
//' @param K An integer, the number of used eigenpairs.
//' @param N A numeric vector with length(m), total count.
//' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
//' the defaulting value is 1e-3.
//'
//' @return `nll` negative log marginal likelihood functional with parameter t.
//' @export
//'
//' @examples
//' Z <- matrix(rnorm(3*3),3,3)
//' X <- Z%*%t(Z)
//' eigenpair <- eigen(X)
//' Y <- sample(c(0,1),3, replace=TRUE)
//' negative_marginal_likelihood_logit(eigenpair, Y, c(1:3), 2)
*/
double negative_marginal_likelihood_logit_cpp(unsigned n, const double *x, double *grad, void * data);

// Create negative log marginal likelihood functional for the regression
double negative_marginal_likelihood_regression_cpp(unsigned n, const double *x, double *grad, void * data);

// Create negative log posterior functional with inverse gamma prior for logistic regression
double negative_log_posterior_logit_cpp(unsigned n, const double *x, double *grad, void *data);

// Create negative log posterior functional with inverse gamma prior for the regression
double negative_log_posterior_regression_cpp(unsigned n, const double *x, double *grad, void *data);


// marginal likelihood objective function data in the binary classification
struct MargOFData {
  const EigenPair & eigenpair;
  const Eigen::VectorXd & Y;
  const Eigen::VectorXd & N;
  const Eigen::VectorXi & idx;
  const int K;
  const double sigma;

  MargOFData(const EigenPair & _eigenpair, const Eigen::VectorXd & _Y, const Eigen::VectorXd & _N, const Eigen::VectorXi & _idx,
         int _K, double _sigma = 1e-3) : eigenpair(_eigenpair), Y(_Y), N(_N), idx(_idx), K(_K), sigma(_sigma) {}
};

// marginal likelihood objective function data in the regression
struct MargOFDataReg {
  const EigenPair & eigenpair;
  const Eigen::VectorXd & Y;
  const Eigen::VectorXi & idx;
  const int K;
  const double sigma;
  MargOFDataReg(const EigenPair & _eigenpair, const Eigen::VectorXd & _Y, const Eigen::VectorXi & _idx,
                int _K, double _sigma = 1e-8) : eigenpair(_eigenpair), Y(_Y), idx(_idx), K(_K), sigma(_sigma) {}
};

// marginal likelihood objective function data in the multi regression
struct MargOFDataMulReg {
  const EigenPair & eigenpair;
  const Eigen::MatrixXd & Y;
  const Eigen::VectorXi & idx;
  const int K;
  const double sigma;
  MargOFDataMulReg(const EigenPair & _eigenpair, const Eigen::MatrixXd & _Y, const Eigen::VectorXi & _idx,
                int _K, double _sigma = 1e-8) : eigenpair(_eigenpair), Y(_Y), idx(_idx), K(_K), sigma(_sigma) {}
};


// posterior objective function data in the binary classification
struct PostOFData {
  const EigenPair & eigenpair;
  const Eigen::VectorXd & Y;
  const Eigen::VectorXd & N;
  const Eigen::VectorXi & idx;
  const int K;
  const double sigma;
  const double p, q, tau;

  PostOFData(const EigenPair & _eigenpair, const Eigen::VectorXd & _Y, const Eigen::VectorXd & _N, const Eigen::VectorXi & _idx,
             int _K, double _sigma = 1e-3, double _p = 1e-2, double _q = 10,
             double _tau = 2) : eigenpair(_eigenpair), Y(_Y), N(_N), idx(_idx), K(_K), sigma(_sigma), p(_p), q(_q), tau(_tau) {}
};

// posterior objective function data in the regression
struct PostOFDataReg {
  const EigenPair & eigenpair;
  const Eigen::VectorXd & Y;
  const Eigen::VectorXi & idx;
  const int K;
  const double sigma;
  const double p, q, tau;
  const double alpha, beta;

  PostOFDataReg(const EigenPair & _eigenpair, const Eigen::VectorXd & _Y,  const Eigen::VectorXi & _idx,
             int _K, double _sigma = 1e-5, double _p = 1, double _q = 10,
             double _tau = 2, double _alpha = 1e-1, double _beta = 1e-3) : eigenpair(_eigenpair), Y(_Y), idx(_idx), K(_K), sigma(_sigma), p(_p), q(_q), tau(_tau), alpha(_alpha), beta(_beta) {}
};

// posterior objective function data in the multi regression
struct PostOFDataMulReg {
  const EigenPair & eigenpair;
  const Eigen::MatrixXd & Y;
  const Eigen::VectorXi & idx;
  const int K;
  const double sigma;
  const double p, q, tau;
  const double alpha, beta;

  PostOFDataMulReg(const EigenPair & _eigenpair, const Eigen::MatrixXd & _Y,  const Eigen::VectorXi & _idx,
                int _K, double _sigma = 1e-5, double _p = 1, double _q = 10,
                double _tau = 2, double _alpha = 1e-1, double _beta = 1e-3) : eigenpair(_eigenpair), Y(_Y), idx(_idx), K(_K), sigma(_sigma), p(_p), q(_q), tau(_tau), alpha(_alpha), beta(_beta) {}
};

// return value, contains optimal parameter t and objective value obj
struct ReturnValue{
  double t;
  double obj;
  ReturnValue(double _t, double _obj) : t(_t), obj(_obj) {}
  ReturnValue() {
    t = 0;
    obj = -std::numeric_limits<double>::infinity();
  }
};

// return value in regression, contains hyper-parameters x = (t, sigma^2) and objective value obj
struct ReturnValueReg{
  std::vector<double> x;
  double obj;
  ReturnValueReg(const std::vector<double>& _x, double _obj) : x(_x), obj(_obj) {}
  ReturnValueReg() {
    obj = -std::numeric_limits<double>::infinity();
  }

};


/*
//' Learn diffusion time t by maximizing log marginal likelihood or log posterior
//'
//' @param eigenpair A list includes values and vectors.
//' @param Y A numeric vector with length(m), count of the positive class.
//' @param idx An integer vector with length(m), the index of training samples.
//' @param K An integer, the number of used eigenpairs.
//' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
//' the defaulting value is 1e-3.
//' @param N A numeric vector with length(m), total count.
//' @param approach A character vector, taking value in c("posterior", "marginal"),
//' decides which objective function to be optimized, defaulting value is `posterior`.
//' @param t0 A positive double, the initial guess for t, defaulting value `10`.
//' @param lower the lower bound on t, defaulting value `1e-3`.
//' @param upper the upper bound on t, defaulting value `Inf`.
//'
//' @return A list with two components
//' \describe{
//' \item{t}{the optimal diffusion time.}
//' \item{obj}{the corresponding optimal objective function value.}
//' }
//' @export
//'
//' @examples
//' X0 <- matrix(rnorm(3*3), 3, 3)
//' X1 <- matrix(rnorm(3*3, 5), 3, 3)
//' Y <- c(1,1,1,0,0,0)
//' X <- rbind(X0,X1)
//' X0_new <- matrix(rnorm(10*3),10,3)
//' X1_new <- matrix(rnorm(10*3, 2),10,3)
//' X_new <- rbind(X0_new, X1_new)
//' s <- 6; r <- 3
//' m <- 6; K <- 3
//' eigenpair <- heat_kernel_spectrum(X, X_new, s, r, K=K)
//' train_lae_logit_gp(eigenpair, Y, c(1:m), K)
*/
ReturnValue train_lae_logit_gp_cpp(void *data, std::string approach = "posterior",
                                   double t0 = -1, double lb = 1e-3, double ub = std::numeric_limits<double>::infinity());



// Learn diffusion time t and noise sigma by maximizing log marginal likelihood or log posterior
ReturnValueReg train_regression_gp_cpp(void *data, std::string approach = "posterior",
                                       std::vector<double>* x0 = nullptr,
                                       std::vector<double>* lb = nullptr, std::vector<double>* ub = nullptr);

#endif
