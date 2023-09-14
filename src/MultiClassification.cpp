// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <vector>
#include <list>

#include "MultiClassification.h"
#include "Utils.h"
#include "Spectrum.h"
#include "train.h"
#include "Predict.h"


// split one multi-classes data set into multiple binary-classes data set
// The multi-classes should be continuous integers, such as "0,1,2,...,9".
std::list<BinaryModel> multi_train_split(const Eigen::VectorXd & Y, int min, int max) {
  int m = Y.size();

  std::list<BinaryModel> models;

  Eigen::VectorXi idx;
  Eigen::VectorXd Yj;
  for(int j=min; j<max; j++) {
    std::vector<int> idx_vec;
    std::vector<double> Yj_vec;
    for(int i=0; i<m; i++) {
      if(Y(i)>=j) {
        idx_vec.push_back(i);
        (Y(i)==j) ? Yj_vec.push_back(1) : Yj_vec.push_back(0);
      }
    }

    idx = Eigen::Map<Eigen::VectorXi>(idx_vec.data(), idx_vec.size());
    Yj =  Eigen::Map<Eigen::VectorXd>(Yj_vec.data(), Yj_vec.size());
    models.push_back(BinaryModel(idx, Yj));
  }

  return models;
}


std::list<BinaryModel> train_logit_mult_gp_cpp(const EigenPair & eigenpair,
                                               const Eigen::VectorXd & Y,
                                               int K, int min, int max,
                                               double sigma, std::string approach) {
  // split train set
  std::list<BinaryModel> models = multi_train_split(Y, min, max);

  // train all binary classification models
  for(auto it=models.begin();it!=models.end();it++) {
    if(approach=="posterior") {
      PostOFData postdata(eigenpair, it->Y, it->N, it->idx, K, sigma);
      it->res = train_lae_logit_gp_cpp(&postdata, approach);
    } else if(approach=="marginal") {
      MargOFData margdata(eigenpair, it->Y, it->N, it->idx, K, sigma);
      it->res = train_lae_logit_gp_cpp(&margdata, approach);
    } else {
      Rcpp::stop("This model selection approach is not supported!");
    }
  }

  return models;
}


Eigen::VectorXd test_logit_mult_gp_cpp(const std::list<BinaryModel> & models,
                              const EigenPair & eigenpair,
                              int m, int m_new,
                              int K, int min, int max,
                              double sigma) {
  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m_new, 0, m_new-1);
  Eigen::VectorXd Y_pred = Eigen::VectorXd::Constant(m_new, max);
  int j = min;
  for(auto it=models.begin();it!=models.end();it++) {
    // construct covariance matrix
    const ReturnValue & res = it->res;
    Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, res.t, it->idx, it->idx);
    Cvv.diagonal().array() += sigma;
    Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, res.t, idx.array()+m, it->idx);

    // predict binary labels on new samples
    Eigen::VectorXd Yj_pred = Rcpp::as<Eigen::VectorXd>(test_pgbinary_cpp(Cvv, it->Y, Cnv)["Y_pred"]);

    // compute the corresponding multi-labels
    std::vector<int> idx_vec;
    int mj_new = idx.size();
    for(int i=0;i<mj_new;i++) {
      if(Yj_pred(i)==1) {
        Y_pred(idx(i)) = j;
      } else {
        idx_vec.push_back(idx(i));
      }
    }
    idx = Eigen::Map<Eigen::VectorXi>(idx_vec.data(), idx_vec.size());

    j++;
  }
  if(j!=max) {
    Rcpp::stop("The number of splitted binary models is wrong!");
  }
  return Y_pred;
}






