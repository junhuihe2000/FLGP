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


// Train Gaussian process logistic multinomial regression
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

// Predict Gaussian process logistic multinomial regression
Eigen::VectorXd predict_logit_mult_gp_cpp(const std::list<BinaryModel> & models,
                                          const EigenPair & eigenpair,
                                          const Eigen::VectorXi & idx,
                                          int K, int min, int max,
                                          double sigma);

#endif
