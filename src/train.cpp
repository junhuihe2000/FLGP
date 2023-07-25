// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "PGLogitModel.h"

using namespace Rcpp;
using namespace Eigen;



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
//[[Rcpp::export(marginal_log_likelihood_logit_la_cpp)]]
double marginal_log_likelihood_logit_la_cpp(const Eigen::MatrixXd & C,
                                                const Eigen::VectorXd & Y,
                                                const Eigen::VectorXd & N,
                                                double tol = 1e-5,
                                                unsigned int max_iter = 100) {
  unsigned int m = Y.rows();
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


