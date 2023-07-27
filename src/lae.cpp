// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include <algorithm>
#include <numeric>

#include "Utils.h"
#include "lae.h"


using namespace Rcpp;
using namespace Eigen;





struct LAE_Parallel : public RcppParallel::Worker {
  // original samples
  const Eigen::MatrixXd & X;
  // reference points
  const Eigen::MatrixXd & U;
  // knn index
  const Eigen::MatrixXi & ind_knn;

  // destination matrix
  Eigen::MatrixXd & Z;

  // column index of U
  Eigen::ArrayXi cols;

  // initialize from Rcpp input and output matrixces (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  LAE_Parallel(const Eigen::MatrixXd & X, const Eigen::MatrixXd & U,
               const Eigen::MatrixXi & ind_knn, Eigen::MatrixXd & Z)
    : X(X), U(U), ind_knn(ind_knn), Z(Z) {
    int d = U.cols();
    cols = Eigen::ArrayXi::LinSpaced(d,0,d-1);
  }

  // function call operator that work for the specified range (begin/end)
  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t i=begin;i<end;i++) {
      Eigen::MatrixXd Ui = mat_indexing(U, ind_knn.row(i), cols);
      Z.row(i) = local_anchor_embedding_cpp(X.row(i), Ui);
    }
  }
};



Eigen::SparseMatrix<double, Eigen::RowMajor> LAE_cpp(const Eigen::MatrixXd & X,
                                                     const Eigen::MatrixXd & U,
                                                     int r) {
  int n = X.rows();
  int s = U.rows();
  Eigen::MatrixXi ind_knn = KNN_cpp(X, U, r)["ind_knn"];
  Eigen::MatrixXd Z(n,r);
  LAE_Parallel lae_parallel(X, U, ind_knn, Z);
  RcppParallel::parallelFor(0, n, lae_parallel);

  Eigen::SparseMatrix<double, Eigen::RowMajor> Z_sp(n,s);
  Z_sp.reserve(Eigen::VectorXi::Constant(n, r));

  for(int i=0;i<n;i++) {
    for(int j=0;j<r;j++) {
      Z_sp.insert(i, ind_knn(i,j)) = Z(i,j);
    }
  }

  return Z_sp;
}





Eigen::RowVectorXd local_anchor_embedding_cpp(const Eigen::RowVectorXd & x,
                                              const Eigen::MatrixXd & U) {
  int r = U.rows();

  // initialize
  Eigen::RowVectorXd z_prev, z_curr;
  z_prev = Eigen::RowVectorXd::Constant(r, 1.0/r); z_curr = z_prev;
  double delta_prev = 0.0; double delta_curr = 1.0;
  double beta_curr = 1.0;
  // stop criterion
  double tol = 1e-5; int T = 100;

  // useful quantities
  Eigen::MatrixXd Ut = U.transpose();
  Eigen::MatrixXd UUt = U*Ut;

  int j;
  double alpha, g_v, beta, g_z, g_tilde;
  Eigen::RowVectorXd v, grad_v, v_tilde, z;

  // Nesterov momentum method
  for(int t=0;t<T;t++) {
    // momentum exponential ratio
    alpha = (delta_prev-1.0)/delta_curr;
    // sliding average
    v = z_curr + alpha*(z_curr-z_prev);
    // value
    g_v = (x-v*U).squaredNorm()/2.0;
    // gradient
    grad_v = v*UUt - x*Ut;
    // backtracking line search
    j = 0;
    while(true) {
      // backtrack
      beta = std::pow(2,j) *beta_curr;
      // gradient descent
      v_tilde = v - 1.0/beta*grad_v;
      // projection
      z = v_to_z_cpp(v_tilde);
      // update condition
      g_z = (x-z*U).squaredNorm()/2.0;
      g_tilde = g_v + grad_v.dot(z-v) + beta*(z-v).squaredNorm()/2.0;
      if(g_z<=g_tilde) {
        beta_curr = beta;
        z_prev = z_curr;
        z_curr = z;
        break;
      }
      j++;
    }
    delta_prev = delta_curr;
    delta_curr = (1.0+std::sqrt(1.0+4.0*delta_curr*delta_curr))/2.0;
    // repeat until convergence
    if((z_curr-z_prev).squaredNorm()<tol) {break;}
  }

  return z_curr;
}



Eigen::RowVectorXd v_to_z_cpp(const Eigen::RowVectorXd & v){
  int r = v.size();
  Eigen::RowVectorXd v_desc = v;
  std::sort(v_desc.data(), v_desc.data()+r, std::greater<double>());
  Eigen::RowVectorXd v_cumsum(r);
  std::partial_sum(v_desc.data(), v_desc.data()+r, v_cumsum.data());
  Eigen::RowVectorXd v_star = v_desc.array() - (v_cumsum.array()-1)/Eigen::ArrayXd::LinSpaced(r, 1, r).transpose();
  int rho;
  for(rho=r;rho>0;rho--) {
    if(v_star(rho-1)>0) {break;}
  }

  double theta = (v_desc.head(rho).sum() - 1.0)/rho;
  Eigen::RowVectorXd z = v.array() - theta;
  std::transform(z.data(), z.data()+r, z.data(), [](double a) {return std::max(a, 0.0);});
  return z;
}
