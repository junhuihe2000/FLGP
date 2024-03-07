#ifndef LAE_H
#define LAE_H


// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <algorithm>
#include <numeric>

#include "PGLogitModel.h"
#include "Utils.h"



//' Local anchor embedding
//'
//' @param X A numeric matrix with dim (n,d), original sample,
//' each row indicates one original point in R^d.
//' @param U A numeric matrix with dim (s,d) or (s,d+1), sub-sample,
//' each row indicates one representative point in R^d,
//' where the d+1 column indicates the number of points in each cluster if it exists.
//' @param r An integer, the number of the nearest neighbor points.
//'
//' @returns A numeric 'sparse' matrix with dim (n,s) and n*r non-zero entries, r << s,
//' the stochastic transition matrix from X to U.
//' @export
//'
//' @examples
//' X <- matrix(rnorm(10*3),10,3)
//' r <- 3
//' U <- matrix(rnorm(5*3),5,3)
//' LAE_cpp(X, U, r)
// [[Rcpp::export(LAE_cpp)]]
Eigen::SparseMatrix<double, Eigen::RowMajor> LAE_cpp(const Eigen::MatrixXd & X,
                                                      const Eigen::MatrixXd & U,
                                                      int r = 3);


//' Local anchor embedding of one single point by
//' gradient descent projection with Nesterov's methods
//'
//' @param x A numeric row vector with length(d), indicates the single point to be embedded.
//' @param U A numeric matrix with dimension (r, d), the columns of U
//' are equal to the length of x, including KNN reference points.
//'
//' @returns A numeric vector with the length r, convex combination coefficients.
// [[Rcpp::export(local_anchor_embedding_cpp)]]
Eigen::RowVectorXd local_anchor_embedding_cpp(const Eigen::RowVectorXd & x,
                                               const Eigen::MatrixXd & U);



//' Simplex projection into convex combination coefficients
//'
//' @param v A numeric vector to be projected.
//'
//' @returns A numeric vector of convex combination coefficients with
//' the same length of v.
// [[Rcpp::export(v_to_z_cpp)]]
Eigen::RowVectorXd v_to_z_cpp(const Eigen::RowVectorXd & v);


#endif
