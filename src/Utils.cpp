// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include <iostream>
#include <exception>

#include "Utils.h"

using namespace Rcpp;
using namespace Eigen;


struct KNN_Index : public RcppParallel::Worker {
  // source matrix
  const Eigen::MatrixXd & input;

  int r;

  int s;

  // destination matrix
  Eigen::MatrixXi & output;

  // initialize from Rcpp input and output matrixces (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  KNN_Index(const Eigen::MatrixXd & input, Eigen::MatrixXi & output, int r) : input(input), output(output), r(r) {
    s = input.cols();
  }

  // function call operator that work for the specified range (begin/end)
  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t i=begin;i<end;i++) {
      Eigen::ArrayXi ind = Eigen::ArrayXi::LinSpaced(s, 0, s-1);
      const Eigen::RowVectorXd & row = input.row(i);
      std::partial_sort(ind.data(), ind.data()+r, ind.data()+s, [&row](int i1, int i2) {return row(i1)<row(i2);});
      output.row(i) = ind.head(r);
    }
  }
};




Rcpp::List KNN_cpp(const Eigen::MatrixXd & X, const Eigen::MatrixXd & U, int r,
                   Rcpp::String distance, bool output) {
  int n = X.rows();
  int s = U.rows();
  Eigen::MatrixXd distances_mat;
  try {
    if(distance=="Euclidean") {
      distances_mat = ((-2*X*U.transpose()).colwise() + X.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
    } else {
      throw std::invalid_argument("The distance method of KNN is not supported!\n");
    }
  }
  catch(const std::invalid_argument& e) {
    std::cout << e.what() << std::endl;
  }

  Eigen::MatrixXi distances_ind(n,r);
  KNN_Index knn_index(distances_mat, distances_ind, r);
  RcppParallel::parallelFor(0, n, knn_index);

  if(!output) {
    return Rcpp::List::create(Named("ind_knn")=distances_ind);
  } else {
    Eigen::SparseMatrix<double, Eigen::RowMajor> distances_sp(n,s);
    distances_sp.reserve(Eigen::VectorXi::Constant(n,r));
    for(int i=0;i<n;i++) {
      for(int j=0;j<r;j++) {
        int indj = distances_ind(i,j);
        distances_sp.insert(i,indj) = distances_mat(i, indj);
      }
    }
    return Rcpp::List::create(Named("ind_knn")=distances_ind, Named("distances_sp")=distances_sp);
  }
}


