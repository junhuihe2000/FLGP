// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// [[Rcpp::depends(BH, bigmemory)]]
#include <bigmemory/BigMatrix.h>

#include "Spectrum.h"
#include "Utils.h"
#include "train.h"
#include "Predict.h"


Eigen::MatrixXd big_subsample_cpp(SEXP X_, const Eigen::MatrixXd & X, int s, std::string method) {
  int n = X.rows(); int d = X.cols();
  Eigen::MatrixXd U;

  if(method=="kmeans") {
    Rcpp::Environment biganalytics = Rcpp::Environment::namespace_env("biganalytics");
    Rcpp::Function bigkmeans = biganalytics["bigkmeans"];
    Rcpp::List cluster_kmeans = bigkmeans(Rcpp::Named("x")=X_,
                                           Rcpp::Named("centers")=s,
                                           Rcpp::Named("iter.max")=100,
                                           Rcpp::Named("nstart")=10);
    U.resize(s, d+1);
    U.leftCols(d) = Rcpp::as<Eigen::MatrixXd>(cluster_kmeans["centers"]);
    U.col(d) =  Rcpp::as<Eigen::VectorXd>(cluster_kmeans["size"]);
  } else if(method=="random") {
    Eigen::VectorXi rows = Rcpp::as<Eigen::VectorXi>(Rcpp::sample(n, s)).array()-1;
    U = mat_indexing(X, rows, Eigen::VectorXi::LinSpaced(d,0,d-1));
  } else {
    Rcpp::stop("The subsample method is not supported!");
  }

  return U;
}


EigenPair big_heat_kernel_spectrum_cpp(SEXP X_, const Eigen::MatrixXd & X_all,
                                   int s, int r, int K, const Rcpp::List & models) {

  Eigen::MatrixXd U = big_subsample_cpp(X_, X_all, s, Rcpp::as<std::string>(models["subsample"]));
  Eigen::SparseMatrix<double, Eigen::RowMajor> Z = cross_similarity_lae_cpp(X_all, U, r, Rcpp::as<std::string>(models["gl"]));

  EigenPair eigenpair = spectrum_from_Z_cpp(Z, K, models["root"]);

  return eigenpair;
}


// [[Rcpp::export(big_fit_lae_logit_gp_cpp)]]
Rcpp::List big_fit_lae_logit_gp_cpp(SEXP big_X_all, Rcpp::NumericVector Y_train,
                                int s, int r, int K, Rcpp::NumericVector N_train,
                                double sigma, std::string approach,
                                Rcpp::List models,
                                bool output_cov) {
  std::cout << "Big binary classification with local anchor embedding:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::VectorXd N(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(N_train));

  Rcpp::XPtr<BigMatrix> xpmat(big_X_all);
  Eigen::Map<Eigen::MatrixXd> X_all = Eigen::Map<Eigen::MatrixXd>((double*)xpmat->matrix(), xpmat->nrow(), xpmat->ncol());

  int m = Y_train.size(); int m_new = X_all.rows() - m;
  int n = m + m_new;

  if(K<0) {
    K = s;
  }

  EigenPair eigenpair = big_heat_kernel_spectrum_cpp(big_X_all, X_all, s, r, K, models);

  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m, 0, m-1);

  // train model
  std::cout << "Training..." << std::endl;
  // empirical Bayes to optimize t
  ReturnValue res;
  if(approach=="posterior") {
    PostOFData postdata(eigenpair, Y, N, idx, K, sigma);
    res = train_lae_logit_gp_cpp(&postdata, approach);
  } else if(approach=="marginal") {
    MargOFData margdata(eigenpair, Y, N, idx, K, sigma);
    res = train_lae_logit_gp_cpp(&margdata, approach);
  } else {
    Rcpp::stop("This model selection approach is not supported!");
  }


  std::cout << "By " << approach << " method, optimal t = " << res.t \
            << ", the objective function is " << res.obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;
  // construct covariance matrix
  Eigen::VectorXi idx0 = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  Eigen::VectorXi idx1 = Eigen::VectorXi::LinSpaced(m_new, m, n-1);
  Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, res.t, idx0, idx0);
  Cvv.diagonal().array() += sigma;
  Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, res.t, idx1, idx0);

  // predict labels on new samples
  Eigen::VectorXd Y_pred = Rcpp::as<Eigen::VectorXd>(test_pgbinary_cpp(Cvv, Y, Cnv)["Y_pred"]);
  std::cout << "Over" << std::endl;

  if(output_cov) {
    Eigen::MatrixXd C(n,m);
    C.topRows(m) = Cvv;
    C.bottomRows(m_new) = Cnv;
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred, Rcpp::Named("C")=C);
  } else {
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred);
  }

}
