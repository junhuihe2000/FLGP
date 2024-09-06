// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <vector>

#include "MultiClassification.h"
#include "Utils.h"
#include "Spectrum.h"
#include "train.h"
#include "Predict.h"



Eigen::MatrixXd multi_train_split(const Eigen::VectorXd & Y) {
  int m = Y.size();
  int J = Y.maxCoeff() + 1;
  Eigen::MatrixXd aug_y = Eigen::MatrixXd::Zero(m,J);

  for(int j=0;j<J;j++) {
    for(int i=0;i<m;i++) {
      if(Y(i)==j) {aug_y(i,j)=1;}
    }
  }

  return aug_y;
}


MultiClassifier train_logit_mult_gp_cpp(const EigenPair & eigenpair,
                                               const Eigen::VectorXd & Y,
                                               const Eigen::VectorXi & idx,
                                               int K,
                                               double sigma, std::string approach) {
  // split train set
  Eigen::MatrixXd aug_y = multi_train_split(Y);
  Eigen::VectorXd N = Eigen::VectorXd::Constant(Y.size(), 1);
  int J = Y.maxCoeff() + 1;
  std::vector<ReturnValue> res_vec;
  // train all binary classification models
  for(int j=0;j<J;j++) {
    if(approach=="posterior") {
      PostOFData postdata(eigenpair, aug_y.col(j), N, idx, K, sigma);
      res_vec.push_back(train_lae_logit_gp_cpp(&postdata, approach));
    } else if(approach=="marginal") {
      MargOFData margdata(eigenpair, aug_y.col(j), N, idx, K, sigma);
      res_vec.push_back(train_lae_logit_gp_cpp(&margdata, approach));
    } else {
      Rcpp::stop("This model selection approach is not supported!");
    }
  }

  return MultiClassifier(aug_y, res_vec);
}



Eigen::VectorXd predict_logit_mult_gp_cpp(const MultiClassifier & multiclassifier,
                                             const EigenPair & eigenpair,
                                             const Eigen::VectorXi & idx,
                                          const Eigen::VectorXi & idx_new,
                                          int K,
                                          double sigma) {
  const Eigen::MatrixXd & aug_y = multiclassifier.aug_y;
  const std::vector<ReturnValue> & res_vec = multiclassifier.res_vec;
  int J = aug_y.cols();
  int m_new = idx_new.rows();
  Eigen::MatrixXd probs(m_new, J);

  for(int j=0;j<J;j++) {
    // construct covariance matrix
    const ReturnValue & res = res_vec[j];
    Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, res.t, idx, idx);
    Cvv.diagonal().array() += sigma;
    Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, res.t, idx_new, idx);

    // predict binary probabilities on new samples
    probs.col(j) = Rcpp::as<Eigen::VectorXd>(test_pgbinary_cpp(Cvv, aug_y.col(j), Cnv, 100, true)["pi_pred"]);
  }

  Eigen::VectorXd y_pred(m_new);
  int loc;
  for(int i=0;i<m_new;i++) {
    probs.row(i).maxCoeff(&loc);
    y_pred(i) = loc;
  }

  return y_pred;
}
