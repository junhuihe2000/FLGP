#ifndef MULTICLASSIFICATION_H
#define MULTICLASSIFICATION_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include "train.h"


/*------------------------------------
 * -----------------------------------
 * -----------------------------------*/

// The multi-classification model consists of multiple binary classification models.
// The basic unit of binary models.
struct BinaryModel{
  Eigen::VectorXi idx;
  Eigen::VectorXd Y;
  Eigen::VectorXd N;
  ReturnValue res;

  BinaryModel(const Eigen::VectorXi & _idx, const Eigen::VectorXd & _Y) : idx(_idx), Y(_Y) {
    N = Eigen::VectorXd::Constant(Y.size(), 1);
  }
  BinaryModel(const Eigen::VectorXi & _idx, const Eigen::VectorXd & _Y, const ReturnValue & _res) : idx(_idx), Y(_Y), res(_res) {
    N = Eigen::VectorXd::Constant(Y.size(), 1);
  }

};

// split one multi-classes data set into multiple binary-classes data set
// The multi-classes should be continuous integers, such as "0,1,2,...,9".
// @param Y A numeric vector with length(m), indicating the labels of multi-classes,
// `Y` should be continuous integers, such as 0,1,2,...,9.
// @param min An integer, indicating the minimum of Y.
// @param max An integer, indicating the maximum of Y.
std::list<BinaryModel> multi_train_split(const Eigen::VectorXd & Y, int min, int max);


/*
//' Train Gaussian process logistic multinomial regression
//'
//' @description Compose J-1 binary logistic regression to implement the multinomial regression.
//'
//' @param eigenpair A list includes values and vectors.
//' @param Y A numeric vector with length(m), each element indicates the label,
//' taking value in `c(0:(J-1))`.
//' @param K An integer, the number of used eigenpairs.
//' @param J An integer, the number of classes.
//' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
//' the defaulting value is 1e-3.
//' @param N A numeric vector with length(m), total count.
//' @param approach A character vector, taking value in c("posterior", "marginal"),
//' decides which objective function to be optimized, defaulting value is `posterior`.
//' @param t0 A positive double, the initial guess for t, defaulting value `10`.
//' @param lower the lower bound on t, defaulting value `1e-3`.
//' @param upper the upper bound on t, defaulting value `Inf`.
//'
//' @return A model list with J-1 model, each model includes four components
//' \describe{
//' \item{Y}{the re-encoded training samples.}
//' \item{idx}{the index of Y in original samples.}
//' \item{t}{the optimal diffusion time.}
//' \item{obj}{the corresponding optimal objective function value.}
//' }
*/
std::list<BinaryModel> train_logit_mult_gp_cpp(const EigenPair & eigenpair,
                                               const Eigen::VectorXd & Y,
                                               int K, int min, int max,
                                               double sigma, std::string approach);



// Test Gaussian process logistic multinomial regression
Eigen::VectorXd test_logit_mult_gp_cpp(const std::list<BinaryModel> & models,
                              const EigenPair & eigenpair,
                              int m, int m_new,
                              int K, int min, int max,
                              double sigma);

#endif
