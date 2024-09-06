#ifndef MULTICLASSIFICATION_H
#define MULTICLASSIFICATION_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <vector>

#include "train.h"


/*------------------------------------
 * -----------------------------------
 * -----------------------------------*/

// The multiclass classification model using one-versus-rest strategies.
struct MultiClassifier{
  Eigen::MatrixXd aug_y;
  std::vector<ReturnValue> res_vec;

  MultiClassifier(const Eigen::MatrixXd & _aug_y,
                  const std::vector<ReturnValue> & _res_vec) : aug_y(_aug_y), res_vec(_res_vec) {}

  MultiClassifier() {}
};

//' split one multi-classes data set into n_classes binary-classes data set using one-versus-rest strategies.
//' The multi-class labels should be continuous integers starting from `0`, such as "0,1,2,...,9".
//' @param Y A numeric vector with length(m), indicating the labels of multi-classes.
//'
//' @return A numeric matrix with dim(m,n_classes).
//'
//' @export
//'
// [[Rcpp::export(multi_train_split)]]
Eigen::MatrixXd multi_train_split(const Eigen::VectorXd & Y);

// Train Gaussian process logistic multinomial regression using one-versus-rest strategies.
MultiClassifier train_logit_mult_gp_cpp(const EigenPair & eigenpair,
                                        const Eigen::VectorXd & Y,
                                        const Eigen::VectorXi & idx,
                                        int K,
                                        double sigma, std::string approach);

// Predict Gaussian process logistic multinomial regression
Eigen::VectorXd predict_logit_mult_gp_cpp(const MultiClassifier & multiclassifier,
                                          const EigenPair & eigenpair,
                                          const Eigen::VectorXi & idx,
                                          const Eigen::VectorXi & idx_new,
                                          int K,
                                          double sigma);


#endif
