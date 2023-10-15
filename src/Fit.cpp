// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
/*
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
*/

#include <iostream>

#include "train.h"
#include "Predict.h"
#include "Spectrum.h"
#include "Utils.h"
#include "MultiClassification.h"
#include "Fit.h"


//-----------------------------------------------------------//
// Gaussian Process regression
//-----------------------------------------------------------//

Rcpp::List fit_lae_regression_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                int s, int r, int K,
                                double sigma, std::string approach, std::string noise,
                                Rcpp::List models,
                                bool output_cov,
                                int nstart) {
  std::cout << "Gaussian regression with local anchor embedding:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  // Map<MatrixXd> fails for Y
  const Eigen::MatrixXd Y(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));

  int m = X.rows(); int m_new = X_new.rows();
  int n = m + m_new;

  if(K<0) {
    K = s;
  }


  EigenPair eigenpair = heat_kernel_spectrum_cpp(X, X_new, s, r, K, models, nstart);

  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m, 0, m-1);

  // train model
  std::cout << "Training..." << std::endl;
  // empirical Bayes to optimize t
  ReturnValueReg res;
  if(approach=="posterior") {
    PostOFDataReg postdatareg(eigenpair, Y, idx, K, sigma);
    res = train_regression_gp_cpp(&postdatareg, approach, noise);
  } else if(approach=="marginal") {
    MargOFDataReg margdatareg(eigenpair, Y, idx, K, sigma);
    res = train_regression_gp_cpp(&margdatareg, approach, noise);
  } else {
    Rcpp::stop("This model selection approach is not supported!");
  }


  std::cout << "By " << approach << " method, optimal t = " << res.x[0] \
            << ", the objective function is " << res.obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;
  // construct covariance matrix
  Eigen::VectorXi idx0 = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  Eigen::VectorXi idx1 = Eigen::VectorXi::LinSpaced(m_new, m, n-1);
  /*
  Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, res.x[0], idx0, idx0);
  Eigen::MatrixXd C_noisy = Cvv;
  C_noisy.diagonal().array() += sigma;
  C_noisy.diagonal().array() += res.x[1];
  Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, res.x[0], idx1, idx0);

  // predict labels on the training set
  Eigen::VectorXd train_pred = test_regression_cpp(C_noisy, Y, Cvv);
  // predict labels on the testing set
  Eigen::VectorXd test_pred = test_regression_cpp(C_noisy, Y, Cnv);
  */

  // predict labels on the training set
  Eigen::MatrixXd train_pred = predict_regression_cpp(eigenpair, Y, idx0, idx0, K, res.x, sigma, noise);
  // predict labels on the testing set
  Eigen::MatrixXd test_pred = predict_regression_cpp(eigenpair, Y, idx0, idx1, K, res.x, sigma, noise);

  Rcpp::List Y_pred = Rcpp::List::create(Rcpp::Named("train")=train_pred,
                                         Rcpp::Named("test")=test_pred);

  std::cout << "Over" << std::endl;

  if(output_cov) {
    Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, res.x[0], idx0, idx0);
    Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, res.x[0], idx1, idx0);

    Eigen::MatrixXd C(n,m);
    C.topRows(m) = Cvv;
    C.bottomRows(m_new) = Cnv;
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred, Rcpp::Named("C")=C,
                              Rcpp::Named("pars")=res.x);
  } else {
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred,
                              Rcpp::Named("pars")=res.x);
  }

}


Rcpp::List fit_se_regression_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                               int s, int r, int K,
                               double sigma, std::vector<double> a2s, std::string approach, std::string noise,
                               Rcpp::List models,
                               bool output_cov,
                               int nstart) {
  std::cout << "Gaussian regression with square exponential kernel:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  // const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::MatrixXd Y(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));

  int m = X.rows(); int m_new = X_new.rows();
  int n = m + m_new; int d = X.cols();

  if(K<0) {
    K = s;
  }

  Eigen::MatrixXd X_all(n, d);
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;
  Eigen::MatrixXd U = subsample_cpp(X_all, s, Rcpp::as<std::string>(models["subsample"]), nstart);
  Rcpp::List res_knn = KNN_cpp(X_all, U.leftCols(d), r, "Euclidean", true);
  const Eigen::MatrixXi& ind_knn = res_knn["ind_knn"];
  const Eigen::SparseMatrix<double, Eigen::RowMajor> & distances_sp = res_knn["distances_sp"];

  double distances_mean = distances_sp.coeffs().sum()/ (n*r);

  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  std::string gl = Rcpp::as<std::string>(models["gl"]);
  bool root = models["root"];
  Eigen::SparseMatrix<double, Eigen::RowMajor> Z = distances_sp;
  int l = a2s.size();


  // train model
  std::cout << "Training..." << std::endl;
  // grid search
  double best_a2 = 0;
  std::vector<double> best_pars;
  double max_obj = -std::numeric_limits<double>::infinity();
  EigenPair best_eigenpair;
  for(int i=0;i<l;i++) {
    double a2 = a2s[i];

    Z.coeffs() = Eigen::exp(-distances_sp.coeffs()/(a2*distances_mean));

    if(gl=="cluster-normalized") {
      graphLaplacian_cpp(Z, gl, U.rightCols(1));
    } else {
      graphLaplacian_cpp(Z, gl);
    }

    EigenPair eigenpair = spectrum_from_Z_cpp(Z, K, root);

    // empirical Bayes to optimize t
    ReturnValueReg res;
    if(approach=="posterior") {
      PostOFDataReg postdatareg(eigenpair, Y, idx, K, sigma);
      res = train_regression_gp_cpp(&postdatareg, approach, noise);
    } else if(approach=="marginal") {
      MargOFDataReg margdatareg(eigenpair, Y, idx, K, sigma);
      res = train_regression_gp_cpp(&margdatareg, approach, noise);
    } else {
      Rcpp::stop("This model selection approach is not supported!");
    }

    if(res.obj>max_obj) {
      max_obj = res.obj;
      best_pars = res.x;
      best_a2 = a2s[i];
      best_eigenpair = eigenpair;
    }
  }

  EigenPair & eigenpair = best_eigenpair;

  std::cout << "By " << approach << " method, optimal epsilon = " << std::sqrt(best_a2) << ", t = " << best_pars[0] \
            << ", sigma = " << sqrt(best_pars[1]) << ", the objective function is " << max_obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;
  // construct covariance matrix
  Eigen::VectorXi idx0 = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  Eigen::VectorXi idx1 = Eigen::VectorXi::LinSpaced(m_new, m, n-1);
  /*
  Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx0, idx0);
  Eigen::MatrixXd C_noisy = Cvv;
  C_noisy.diagonal().array() += sigma + best_pars[1];
  Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx1, idx0);

  // predict labels on the training set
  Eigen::VectorXd train_pred = test_regression_cpp(C_noisy, Y, Cvv);
  // predict labels on the testing set
  Eigen::VectorXd test_pred = test_regression_cpp(C_noisy, Y, Cnv);
  */

  // predict labels on the training set
  Eigen::MatrixXd train_pred = predict_regression_cpp(eigenpair, Y, idx0, idx0, K, best_pars, sigma, noise);
  // predict labels on the testing set
  Eigen::MatrixXd test_pred = predict_regression_cpp(eigenpair, Y, idx0, idx1, K, best_pars, sigma, noise);

  Rcpp::List Y_pred = Rcpp::List::create(Rcpp::Named("train")=train_pred,
                                         Rcpp::Named("test")=test_pred);

  std::cout << "Over" << std::endl;

  if(output_cov) {
    Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx0, idx0);
    Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx1, idx0);

    Eigen::MatrixXd C(n,m);
    C.topRows(m) = Cvv;
    C.bottomRows(m_new) = Cnv;
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred, Rcpp::Named("C")=C,
                              Rcpp::Named("pars")=best_pars);
  } else {
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred,
                              Rcpp::Named("pars")=best_pars);
  }
}

Rcpp::List fit_nystrom_regression_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                    int s, int K,
                                    double sigma, std::vector<double> a2s, std::string approach, std::string noise,
                                    Rcpp::List models,
                                    bool output_cov,
                                    int nstart) {
  std::cout << "Gaussian regression with Nystrom extension:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  // const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::MatrixXd Y(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));

  int m = X.rows(); int m_new = X_new.rows(); int n = m + m_new;
  int d = X.cols();

  Eigen::MatrixXd X_all(n,d);
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;

  const std::string subsample = Rcpp::as<std::string>(models["subsample"]);

  const Eigen::MatrixXd U = subsample_cpp(X_all, s, subsample, nstart).leftCols(d);

  const Eigen::MatrixXd distances_UU  = ((-2*U*U.transpose()).colwise() + U.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
  const Eigen::MatrixXd distances_allU = ((-2*X_all*U.transpose()).colwise() + X_all.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
  const Eigen::MatrixXd & distances_XU = distances_allU.topRows(m);

  double distances_mean = distances_UU.array().sum()/(s*s);

  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m,0,m-1);

  // train model
  std::cout << "Training..." << std::endl;
  // grid search
  double best_a2 = 0;
  std::vector<double> best_pars;
  double max_obj = -std::numeric_limits<double>::infinity();
  EigenPair best_eigenpair;
  Eigen::MatrixXd best_Z_UU;
  int l = a2s.size();

  Rcpp::Environment RSpectra = Rcpp::Environment::namespace_env("RSpectra");
  Rcpp::Function eigs_sym = RSpectra["eigs_sym"];

  for(int i=0;i<l;i++) {
    double a2 = a2s[i];

    Eigen::MatrixXd Z_UU = Eigen::exp(-distances_UU.array()/(a2*distances_mean));
    Eigen::VectorXd Z_UU_rowsums = Z_UU.rowwise().sum().array()+1e-5;
    Eigen::MatrixXd A_UU = (1.0/Z_UU_rowsums.array()).matrix().asDiagonal()*Z_UU*(1.0/Z_UU_rowsums.array()).matrix().asDiagonal();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> sqrt_D_inv = (1.0/(A_UU.rowwise().sum().array()+1e-5).sqrt()).matrix().asDiagonal();
    Eigen::MatrixXd W_UU = sqrt_D_inv*A_UU*sqrt_D_inv;


    Rcpp::List res_eigs = eigs_sym(Rcpp::Named("A")=W_UU,
                                   Rcpp::Named("k")=K);
    EigenPair eigenpair_UU(Rcpp::as<Eigen::VectorXd>(res_eigs["values"]),
                           Rcpp::as<Eigen::MatrixXd>(res_eigs["vectors"]));
    eigenpair_UU.vectors = std::sqrt(s)*sqrt_D_inv*eigenpair_UU.vectors;

    // Nystrom extension formula
    Eigen::MatrixXd Z_XU = Eigen::exp(-distances_XU.array()/(a2*distances_mean));
    Eigen::VectorXd Z_XU_rowsums = Z_XU.rowwise().sum().array()+1e-5;
    Eigen::MatrixXd A_XU = (1.0/Z_XU_rowsums.array()).matrix().asDiagonal()*Z_XU*(1.0/Z_UU_rowsums.array()).matrix().asDiagonal();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> D_inv = (1.0/(A_XU.rowwise().sum().array()+1e-5)).matrix().asDiagonal();
    Eigen::MatrixXd W_XU = D_inv*A_XU;
    EigenPair eigenpair = eigenpair_UU;
    eigenpair.vectors = W_XU*eigenpair.vectors*(1.0/eigenpair.values.array()).matrix().asDiagonal();

    // empirical Bayes to optimize t
    ReturnValueReg res;
    if(approach=="posterior") {
      PostOFDataReg postdatareg(eigenpair, Y, idx, K, sigma);
      res = train_regression_gp_cpp(&postdatareg, approach, noise);
    } else if(approach=="marginal") {
      MargOFDataReg margdatareg(eigenpair, Y, idx, K, sigma);
      res = train_regression_gp_cpp(&margdatareg, approach, noise);
    } else {
      Rcpp::stop("This model selection approach is not supported!");
    }

    if(res.obj>max_obj) {
      max_obj = res.obj;
      best_pars = res.x;
      best_a2 = a2s[i];
      best_eigenpair = eigenpair_UU;
      best_Z_UU = Z_UU;
    }

  }

  std::cout << "By " << approach << " method, optimal epsilon = " << std::sqrt(best_a2) << ", t = " << best_pars[0] \
            << ", sigma = " << sqrt(best_pars[1]) << ", the objective function is " << max_obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;
  // Nystrom extension
  Eigen::MatrixXd Z_allU = Eigen::exp(-distances_allU.array()/(best_a2*distances_mean));
  Eigen::MatrixXd A_allU = (1.0/Z_allU.rowwise().sum().array()).matrix().asDiagonal()*Z_allU*(1.0/(best_Z_UU.colwise().sum().array())).matrix().asDiagonal();
  Eigen::MatrixXd W_allU = (1.0/A_allU.rowwise().sum().array()).matrix().asDiagonal()*A_allU;
  EigenPair & eigenpair = best_eigenpair;
  eigenpair.vectors = W_allU*eigenpair.vectors*(1.0/eigenpair.values.array()).matrix().asDiagonal();
  // construct covariance matrix
  Eigen::VectorXi idx0 = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  Eigen::VectorXi idx1 = Eigen::VectorXi::LinSpaced(m_new, m, n-1);
  /*
  Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx0, idx0);
  Eigen::MatrixXd C_noisy = Cvv;
  C_noisy.diagonal().array() += sigma + best_pars[1];
  Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx1, idx0);

  // predict labels on the training set
  Eigen::VectorXd train_pred = test_regression_cpp(C_noisy, Y, Cvv);
  // predict labels on the testing set
  Eigen::VectorXd test_pred = test_regression_cpp(C_noisy, Y, Cnv);
  */

  // predict labels on the training set
  Eigen::MatrixXd train_pred = predict_regression_cpp(eigenpair, Y, idx0, idx0, K, best_pars, sigma, noise);
  // predict labels on the testing set
  Eigen::MatrixXd test_pred = predict_regression_cpp(eigenpair, Y, idx0, idx1, K, best_pars, sigma, noise);

  Rcpp::List Y_pred = Rcpp::List::create(Rcpp::Named("train")=train_pred,
                                         Rcpp::Named("test")=test_pred);

  std::cout << "Over" << std::endl;

  if(output_cov) {
    Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx0, idx0);
    Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx1, idx0);

    Eigen::MatrixXd C(n,m);
    C.topRows(m) = Cvv;
    C.bottomRows(m_new) = Cnv;
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred, Rcpp::Named("C")=C,
                              Rcpp::Named("pars")=best_pars);
  } else {
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred,
                              Rcpp::Named("pars")=best_pars);
  }
}


Rcpp::List fit_gl_regression_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                               int K,
                               double sigma, std::vector<double> a2s,
                               double threshold, bool sparse,
                               std::string approach, std::string noise,
                               Rcpp::List models,
                               bool output_cov) {
  std::cout << "Gaussian regression with graph Laplacian Gaussian process:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  // const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::MatrixXd Y(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));

  int m = X.rows(); int m_new = X_new.rows(); int n = m + m_new;
  int d = X.cols();

  Eigen::MatrixXd X_all(n,d);
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;

  const std::string subsample = Rcpp::as<std::string>(models["subsample"]);


  Eigen::MatrixXd distances;
  double distances_mean;
  Eigen::SparseMatrix<double, Eigen::RowMajor> distances_sp;

  if(sparse) {
    int r = std::max((int)std::round(threshold*n),3); // r will be greater than 3.
    Rcpp::List res_knn = KNN_cpp(X_all, X_all, r, "Euclidean", true);
    distances_sp = res_knn["distances_sp"];
    distances_mean = distances_sp.coeffs().sum()/(n*r) * r;
  } else {
    distances = ((-2.0*X_all*X_all.transpose()).colwise() + X_all.rowwise().squaredNorm()).rowwise() + X_all.rowwise().squaredNorm().transpose();
    distances_mean = distances.array().mean();
  }



  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m,0,m-1);

  // train model
  std::cout << "Training..." << std::endl;
  // grid search
  double best_a2 = 0;
  std::vector<double> best_pars;
  double max_obj = -std::numeric_limits<double>::infinity();
  EigenPair best_eigenpair;
  int l = a2s.size();

  Rcpp::Environment RSpectra = Rcpp::Environment::namespace_env("RSpectra");
  Rcpp::Function eigs_sym = RSpectra["eigs_sym"];

  for(int i=0;i<l;i++) {
    double a2 = a2s[i];
    EigenPair eigenpair;
    if(sparse) {
      Eigen::SparseMatrix<double> Z(distances_sp);
      Z.coeffs() = Eigen::exp(-Z.coeffs()/(a2*distances_mean));
      Z = (Z + Eigen::SparseMatrix<double>(Z.transpose()))/2.0;
      Eigen::VectorXd Z_rowsum = Z*Eigen::VectorXd::Ones(n);
      Eigen::SparseMatrix<double> A = (1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal()*Z*(1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal();
      Eigen::VectorXd sqrt_D_inv = 1.0/((A*Eigen::VectorXd::Ones(n)).array()+1e-5).sqrt();
      Eigen::SparseMatrix<double> W = sqrt_D_inv.asDiagonal()*A*sqrt_D_inv.asDiagonal();

      Rcpp::List res_eigs = eigs_sym(Rcpp::Named("A")=W,
                                     Rcpp::Named("k")=K);
      eigenpair = EigenPair(Rcpp::as<Eigen::VectorXd>(res_eigs["values"]),
                            Rcpp::as<Eigen::MatrixXd>(res_eigs["vectors"]));
      eigenpair.vectors = std::sqrt(n)*sqrt_D_inv.asDiagonal()*eigenpair.vectors;
    } else {
      Eigen::MatrixXd Z = Eigen::exp(-distances.array()/(a2*distances_mean));

      Eigen::VectorXd Z_rowsum = Z*Eigen::VectorXd::Ones(n);
      Eigen::MatrixXd A = (1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal()*Z*(1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal();
      Eigen::VectorXd sqrt_D_inv = 1.0/((A*Eigen::VectorXd::Ones(n)).array()+1e-5).sqrt();
      Eigen::MatrixXd W = sqrt_D_inv.asDiagonal()*A*sqrt_D_inv.asDiagonal();

      Rcpp::List res_eigs = eigs_sym(Rcpp::Named("A")=W,
                                     Rcpp::Named("k")=K);
      eigenpair = EigenPair(Rcpp::as<Eigen::VectorXd>(res_eigs["values"]),
                            Rcpp::as<Eigen::MatrixXd>(res_eigs["vectors"]));
      eigenpair.vectors = std::sqrt(n)*sqrt_D_inv.asDiagonal()*eigenpair.vectors;
    }

    // empirical Bayes to optimize t
    ReturnValueReg res;
    if(approach=="posterior") {
      PostOFDataReg postdatareg(eigenpair, Y, idx, K, sigma);
      res = train_regression_gp_cpp(&postdatareg, approach, noise);
    } else if(approach=="marginal") {
      MargOFDataReg margdatareg(eigenpair, Y, idx, K, sigma);
      res = train_regression_gp_cpp(&margdatareg, approach, noise);
    } else {
      Rcpp::stop("This model selection approach is not supported!");
    }

    if(res.obj>max_obj) {
      max_obj = res.obj;
      best_pars = res.x;
      best_a2 = a2s[i];
      best_eigenpair = eigenpair;
    }

  }

  EigenPair & eigenpair = best_eigenpair;

  std::cout << "By " << approach << " method, optimal epsilon = " << std::sqrt(best_a2) << ", t = " << best_pars[0] \
            << ", sigma = " << sqrt(best_pars[1]) << ", the objective function is " << max_obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;
  // construct covariance matrix
  Eigen::VectorXi idx0 = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  Eigen::VectorXi idx1 = Eigen::VectorXi::LinSpaced(m_new, m, n-1);
  /*
  Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx0, idx0);
  Eigen::MatrixXd C_noisy = Cvv;
  C_noisy.diagonal().array() += sigma + best_pars[1];
  Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx1, idx0);


  // predict labels on the training set
  Eigen::VectorXd train_pred = test_regression_cpp(C_noisy, Y, Cvv);
  // predict labels on the testing set
  Eigen::VectorXd test_pred = test_regression_cpp(C_noisy, Y, Cnv);
  */

  // predict labels on the training set
  Eigen::MatrixXd train_pred = predict_regression_cpp(eigenpair, Y, idx0, idx0, K, best_pars, sigma, noise);
  // predict labels on the testing set
  Eigen::MatrixXd test_pred = predict_regression_cpp(eigenpair, Y, idx0, idx1, K, best_pars, sigma, noise);

  Rcpp::List Y_pred = Rcpp::List::create(Rcpp::Named("train")=train_pred,
                                         Rcpp::Named("test")=test_pred);

  std::cout << "Over" << std::endl;

  if(output_cov) {
    Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx0, idx0);
    Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, best_pars[0], idx1, idx0);

    Eigen::MatrixXd C(n,m);
    C.topRows(m) = Cvv;
    C.bottomRows(m_new) = Cnv;
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred, Rcpp::Named("C")=C,
                              Rcpp::Named("pars")=best_pars);
  } else {
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred,
                              Rcpp::Named("pars")=best_pars);
  }

}



//-----------------------------------------------------------//
// Gaussian Process Classification
//-----------------------------------------------------------//


Rcpp::List fit_lae_logit_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                int s, int r, int K, Rcpp::NumericVector N_train,
                                double sigma, std::string approach,
                                Rcpp::List models,
                                bool output_cov,
                                int nstart) {
  std::cout << "Binary classification with local anchor embedding:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));
  const Eigen::VectorXd N(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(N_train));

  int m = X.rows(); int m_new = X_new.rows();
  int n = m + m_new;

  if(K<0) {
    K = s;
  }


  EigenPair eigenpair = heat_kernel_spectrum_cpp(X, X_new, s, r, K, models, nstart);

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



Rcpp::List fit_lae_logit_mult_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                int s, int r, int K,
                                double sigma, std::string approach,
                                Rcpp::List models,
                                int nstart) {
  std::cout <<"Multinomial classification with local anchor embedding:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));

  int m = X.rows(); int m_new = X_new.rows();

  if(K<0) {
    K = s;
  }

  int min, max;
  min = Y.minCoeff(); max = Y.maxCoeff();

  EigenPair eigenpair = heat_kernel_spectrum_cpp(X, X_new, s, r, K, models, nstart);

  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m, 0, m-1);

  // train model
  std::cout << "Training..." << std::endl;
  // empirical Bayes to optimize t
  std::list<BinaryModel> multi_models = train_logit_mult_gp_cpp(eigenpair, Y, K, min, max, sigma, approach);


  double obj = 0;
  for(auto it=multi_models.begin();it!=multi_models.end();it++) {
    obj += (it->res).obj;
  }
  std::cout << "By " << approach << " method, the optimal objective function is " << obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;

  // predict labels on new samples
  Eigen::VectorXd Y_pred = test_logit_mult_gp_cpp(multi_models, eigenpair, m, m_new, K, min, max, sigma);
  std::cout << "Over" << std::endl;

  return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred);
}




/* RcppParallel often fails with RSpectra, so we choose sequential computing finally
// parallel square exponential kernel
struct SEKPAR : public RcppParallel::Worker {
  // input data
  const Eigen::VectorXd & Y, N;
  const Eigen::MatrixXd & U;
  const Eigen::VectorXi & idx;
  const Eigen::SparseMatrix<double, Eigen::RowMajor> & distances_sp;
  std::string gl, approach;
  bool root;
  double distances_mean, sigma;
  int K;
  // parameters
  const std::vector<double>& a2s;

  // output data
  std::vector<ReturnValue> & rvs;

  // initialize from Rcpp input and output matrixces (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  SEKPAR(const Eigen::VectorXd & Y, const Eigen::VectorXd & N, const Eigen::MatrixXd& U, const Eigen::VectorXi & idx, const Eigen::SparseMatrix<double, Eigen::RowMajor> & distances_sp,
      std::string& gl, std::string& approach, bool root, double distances_mean, double sigma, int K,
      const std::vector<double>& a2s, std::vector<ReturnValue> & rvs) : Y(Y), N(N), U(U), idx(idx), distances_sp(distances_sp), gl(gl),\
      approach(approach), root(root), distances_mean(distances_mean), sigma(sigma), K(K), a2s(a2s), rvs(rvs) {}

  // function call operator that work for the specified range (begin/end)
  void operator()(std::size_t begin, std::size_t end) {

    for(std::size_t i=begin;i<end;i++) {
      double a2 = a2s[i];

      Eigen::SparseMatrix<double, Eigen::RowMajor> Z = distances_sp;
      Z.coeffs() = Eigen::exp(-Z.coeffs()/(a2*distances_mean));

      if(gl=="cluster-normalized") {
        graphLaplacian_cpp(Z, gl, U.rightCols(1));
      } else {
        graphLaplacian_cpp(Z, gl);
      }

      EigenPair eigenpair = spectrum_from_Z_cpp(Z, K, root);

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

      rvs[i] = res;
    }
  }
};
*/



Rcpp::List fit_se_logit_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                               int s, int r, int K, Rcpp::NumericVector N_train,
                               double sigma, std::vector<double> a2s, std::string approach,
                               Rcpp::List models,
                               bool output_cov,
                               int nstart) {
  std::cout << "Binary classification with square exponential kernel:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));
  const Eigen::VectorXd N(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(N_train));

  int m = X.rows(); int m_new = X_new.rows();
  int n = m + m_new; int d = X.cols();

  if(K<0) {
    K = s;
  }

  Eigen::MatrixXd X_all(n, d);
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;
  Eigen::MatrixXd U = subsample_cpp(X_all, s, Rcpp::as<std::string>(models["subsample"]), nstart);
  Rcpp::List res_knn = KNN_cpp(X_all, U.leftCols(d), r, "Euclidean", true);
  const Eigen::MatrixXi& ind_knn = res_knn["ind_knn"];
  const Eigen::SparseMatrix<double, Eigen::RowMajor> & distances_sp = res_knn["distances_sp"];

  double distances_mean = distances_sp.coeffs().sum()/ (n*r);

  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  std::string gl = Rcpp::as<std::string>(models["gl"]);
  bool root = models["root"];
  Eigen::SparseMatrix<double, Eigen::RowMajor> Z = distances_sp;
  int l = a2s.size();

  /* parallel leads to stack imbalance and rstudio restarts
  std::vector<ReturnValue> rvs(l);
  SEKPAR sekpar(Y, N, U, idx, distances_sp, gl, approach, root, distances_mean, sigma,
                K, a2s, rvs);

  RcppParallel::parallelFor(0, l, sekpar);
  */

  // train model
  std::cout << "Training..." << std::endl;
  // grid search
  double best_t = 0, best_a2 = 0;
  double max_obj = -std::numeric_limits<double>::infinity();
  EigenPair best_eigenpair;
  for(int i=0;i<l;i++) {
    double a2 = a2s[i];

    Z.coeffs() = Eigen::exp(-distances_sp.coeffs()/(a2*distances_mean));

    if(gl=="cluster-normalized") {
      graphLaplacian_cpp(Z, gl, U.rightCols(1));
    } else {
      graphLaplacian_cpp(Z, gl);
    }

    EigenPair eigenpair = spectrum_from_Z_cpp(Z, K, root);

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

    if(res.obj>max_obj) {
      max_obj = res.obj;
      best_t = res.t;
      best_a2 = a2s[i];
      best_eigenpair = eigenpair;
    }
  }

  EigenPair & eigenpair = best_eigenpair;

  std::cout << "By " << approach << " method, optimal epsilon = " << std::sqrt(best_a2) << ", t = " << best_t \
            << ", the objective function is " << max_obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;
  // construct covariance matrix
  Eigen::VectorXi idx0 = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  Eigen::VectorXi idx1 = Eigen::VectorXi::LinSpaced(m_new, m, n-1);
  Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, best_t, idx0, idx0);
  Cvv.diagonal().array() += sigma;
  Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, best_t, idx1, idx0);

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



Rcpp::List fit_se_logit_mult_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                               int s, int r, int K,
                               double sigma, std::vector<double> a2s, std::string approach,
                               Rcpp::List models,
                               int nstart) {
  std::cout << "Multinomial classification with square exponential kernel:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));

  int m = X.rows(); int m_new = X_new.rows();
  int n = m + m_new; int d = X.cols();

  if(K<0) {
    K = s;
  }

  int min = Y.minCoeff();
  int max = Y.maxCoeff();

  Eigen::MatrixXd X_all(n, d);
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;
  Eigen::MatrixXd U = subsample_cpp(X_all, s, Rcpp::as<std::string>(models["subsample"]), nstart);
  Rcpp::List res_knn = KNN_cpp(X_all, U.leftCols(d), r, "Euclidean", true);
  const Eigen::MatrixXi& ind_knn = res_knn["ind_knn"];
  const Eigen::SparseMatrix<double, Eigen::RowMajor> & distances_sp = res_knn["distances_sp"];

  double distances_mean = distances_sp.coeffs().sum()/ (n*r);

  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  std::string gl = Rcpp::as<std::string>(models["gl"]);
  bool root = models["root"];
  Eigen::SparseMatrix<double, Eigen::RowMajor> Z = distances_sp;
  int l = a2s.size();

  // train model
  std::cout << "Training..." << std::endl;
  // grid search
  double best_a2 = 0;
  std::list<BinaryModel> best_multi_models;
  double max_obj = -std::numeric_limits<double>::infinity();
  EigenPair best_eigenpair;
  for(int i=0;i<l;i++) {
    double a2 = a2s[i];

    Z.coeffs() = Eigen::exp(-distances_sp.coeffs()/(a2*distances_mean));

    if(gl=="cluster-normalized") {
      graphLaplacian_cpp(Z, gl, U.rightCols(1));
    } else {
      graphLaplacian_cpp(Z, gl);
    }

    EigenPair eigenpair = spectrum_from_Z_cpp(Z, K, root);

    // empirical Bayes to optimize t
    std::list<BinaryModel> multi_models = train_logit_mult_gp_cpp(eigenpair, Y, K, min, max, sigma, approach);

    double obj = 0;
    for(auto it=multi_models.begin();it!=multi_models.end();it++) {
      obj += (it->res).obj;
    }

    if(obj>max_obj) {
      max_obj = obj;
      best_multi_models = multi_models;
      best_a2 = a2s[i];
      best_eigenpair = eigenpair;
    }
  }

  EigenPair & eigenpair = best_eigenpair;
  std::list<BinaryModel> & multi_models = best_multi_models;

  std::cout << "By " << approach << " method, optimal epsilon = " << std::sqrt(best_a2)  \
            << ", the objective function is " << max_obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;

  // predict labels on new samples
  Eigen::VectorXd Y_pred = test_logit_mult_gp_cpp(multi_models, eigenpair, m, m_new, K, min, max, sigma);
  std::cout << "Over" << std::endl;

  return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred);
}




Rcpp::List fit_nystrom_logit_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                    int s, int K, Rcpp::NumericVector N_train,
                                    double sigma, std::vector<double> a2s, std::string approach,
                                    Rcpp::List models,
                                    bool output_cov,
                                    int nstart) {
  std::cout << "Binary classificaiton with Nystrom extension:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));
  const Eigen::VectorXd N(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(N_train));

  int m = X.rows(); int m_new = X_new.rows(); int n = m + m_new;
  int d = X.cols();

  Eigen::MatrixXd X_all(n,d);
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;

  const std::string subsample = Rcpp::as<std::string>(models["subsample"]);

  const Eigen::MatrixXd U = subsample_cpp(X_all, s, subsample, nstart).leftCols(d);

  const Eigen::MatrixXd distances_UU  = ((-2*U*U.transpose()).colwise() + U.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
  const Eigen::MatrixXd distances_allU = ((-2*X_all*U.transpose()).colwise() + X_all.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
  const Eigen::MatrixXd & distances_XU = distances_allU.topRows(m);

  double distances_mean = distances_UU.array().sum()/(s*s);

  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m,0,m-1);

  // train model
  std::cout << "Training..." << std::endl;
  // grid search
  double best_t = 0, best_a2 = 0;
  double max_obj = -std::numeric_limits<double>::infinity();
  EigenPair best_eigenpair;
  Eigen::MatrixXd best_Z_UU;
  int l = a2s.size();

  Rcpp::Environment RSpectra = Rcpp::Environment::namespace_env("RSpectra");
  Rcpp::Function eigs_sym = RSpectra["eigs_sym"];

  for(int i=0;i<l;i++) {
    double a2 = a2s[i];

    Eigen::MatrixXd Z_UU = Eigen::exp(-distances_UU.array()/(a2*distances_mean));
    Eigen::VectorXd Z_UU_rowsums = Z_UU.rowwise().sum().array()+1e-5;
    Eigen::MatrixXd A_UU = (1.0/Z_UU_rowsums.array()).matrix().asDiagonal()*Z_UU*(1.0/Z_UU_rowsums.array()).matrix().asDiagonal();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> sqrt_D_inv = (1.0/(A_UU.rowwise().sum().array()+1e-5).sqrt()).matrix().asDiagonal();
    Eigen::MatrixXd W_UU = sqrt_D_inv*A_UU*sqrt_D_inv;


    Rcpp::List res_eigs = eigs_sym(Rcpp::Named("A")=W_UU,
                                   Rcpp::Named("k")=K);
    EigenPair eigenpair_UU(Rcpp::as<Eigen::VectorXd>(res_eigs["values"]),
                           Rcpp::as<Eigen::MatrixXd>(res_eigs["vectors"]));
    eigenpair_UU.vectors = std::sqrt(s)*sqrt_D_inv*eigenpair_UU.vectors;

    // Nystrom extension formula
    Eigen::MatrixXd Z_XU = Eigen::exp(-distances_XU.array()/(a2*distances_mean));
    Eigen::VectorXd Z_XU_rowsums = Z_XU.rowwise().sum().array()+1e-5;
    Eigen::MatrixXd A_XU = (1.0/Z_XU_rowsums.array()).matrix().asDiagonal()*Z_XU*(1.0/Z_UU_rowsums.array()).matrix().asDiagonal();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> D_inv = (1.0/(A_XU.rowwise().sum().array()+1e-5)).matrix().asDiagonal();
    Eigen::MatrixXd W_XU = D_inv*A_XU;
    EigenPair eigenpair = eigenpair_UU;
    eigenpair.vectors = W_XU*eigenpair.vectors*(1.0/eigenpair.values.array()).matrix().asDiagonal();

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

    if(res.obj>max_obj) {
      max_obj = res.obj;
      best_t = res.t;
      best_a2 = a2s[i];
      best_eigenpair = eigenpair_UU;
      best_Z_UU = Z_UU;
    }

  }

  std::cout << "By " << approach << " method, optimal epsilon = " << std::sqrt(best_a2) << ", t = " << best_t \
            << ", the objective function is " << max_obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;
  // Nystrom extension
  Eigen::MatrixXd Z_allU = Eigen::exp(-distances_allU.array()/(best_a2*distances_mean));
  Eigen::MatrixXd A_allU = (1.0/Z_allU.rowwise().sum().array()).matrix().asDiagonal()*Z_allU*(1.0/(best_Z_UU.colwise().sum().array())).matrix().asDiagonal();
  Eigen::MatrixXd W_allU = (1.0/A_allU.rowwise().sum().array()).matrix().asDiagonal()*A_allU;
  EigenPair & eigenpair = best_eigenpair;
  eigenpair.vectors = W_allU*eigenpair.vectors*(1.0/eigenpair.values.array()).matrix().asDiagonal();
  // construct covariance matrix
  Eigen::VectorXi idx0 = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  Eigen::VectorXi idx1 = Eigen::VectorXi::LinSpaced(m_new, m, n-1);
  Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, best_t, idx0, idx0);
  Cvv.diagonal().array() += sigma;
  Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, best_t, idx1, idx0);

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




Rcpp::List fit_nystrom_logit_mult_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                    int s, int K,
                                    double sigma, std::vector<double> a2s, std::string approach,
                                    Rcpp::List models,
                                    int nstart) {
  std::cout << "Multinomial classification with Nystrom extension:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));

  int m = X.rows(); int m_new = X_new.rows(); int n = m + m_new;
  int d = X.cols();

  int min = Y.minCoeff();
  int max = Y.maxCoeff();

  Eigen::MatrixXd X_all(n,d);
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;

  const std::string subsample = Rcpp::as<std::string>(models["subsample"]);

  const Eigen::MatrixXd U = subsample_cpp(X_all, s, subsample, nstart).leftCols(d);

  const Eigen::MatrixXd distances_UU  = ((-2*U*U.transpose()).colwise() + U.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
  // const Eigen::MatrixXd distances_XU = ((-2*X*U.transpose()).colwise() + X.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
  const Eigen::MatrixXd distances_allU = ((-2*X_all*U.transpose()).colwise() + X_all.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
  const Eigen::MatrixXd & distances_XU = distances_allU.topRows(m);

  double distances_mean = distances_UU.array().sum()/(s*s);

  // train model
  std::cout << "Training..." << std::endl;
  // grid search
  double best_a2 = 0;
  std::list<BinaryModel> best_multi_models;
  double max_obj = -std::numeric_limits<double>::infinity();
  EigenPair best_eigenpair;
  Eigen::MatrixXd best_Z_UU;
  int l = a2s.size();

  Rcpp::Environment RSpectra = Rcpp::Environment::namespace_env("RSpectra");
  Rcpp::Function eigs_sym = RSpectra["eigs_sym"];

  for(int i=0;i<l;i++) {
    double a2 = a2s[i];

    Eigen::MatrixXd Z_UU = Eigen::exp(-distances_UU.array()/(a2*distances_mean));
    Eigen::VectorXd Z_UU_rowsums = Z_UU.rowwise().sum().array()+1e-5;
    Eigen::MatrixXd A_UU = (1.0/Z_UU_rowsums.array()).matrix().asDiagonal()*Z_UU*(1.0/Z_UU_rowsums.array()).matrix().asDiagonal();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> sqrt_D_inv = (1.0/(A_UU.rowwise().sum().array()+1e-5).sqrt()).matrix().asDiagonal();
    Eigen::MatrixXd W_UU = sqrt_D_inv*A_UU*sqrt_D_inv;


    Rcpp::List res_eigs = eigs_sym(Rcpp::Named("A")=W_UU,
                                   Rcpp::Named("k")=K);
    EigenPair eigenpair_UU(Rcpp::as<Eigen::VectorXd>(res_eigs["values"]),
                           Rcpp::as<Eigen::MatrixXd>(res_eigs["vectors"]));
    eigenpair_UU.vectors = std::sqrt(s)*sqrt_D_inv*eigenpair_UU.vectors;

    // Nystrom extension formula
    Eigen::MatrixXd Z_XU = Eigen::exp(-distances_XU.array()/(a2*distances_mean));
    Eigen::VectorXd Z_XU_rowsums = Z_XU.rowwise().sum().array()+1e-5;
    Eigen::MatrixXd A_XU = (1.0/Z_XU_rowsums.array()).matrix().asDiagonal()*Z_XU*(1.0/Z_UU_rowsums.array()).matrix().asDiagonal();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> D_inv = (1.0/(A_XU.rowwise().sum().array()+1e-5)).matrix().asDiagonal();
    Eigen::MatrixXd W_XU = D_inv*A_XU;
    EigenPair eigenpair = eigenpair_UU;
    eigenpair.vectors = W_XU*eigenpair.vectors*(1.0/eigenpair.values.array()).matrix().asDiagonal();

    // empirical Bayes to optimize t
    std::list<BinaryModel> multi_models = train_logit_mult_gp_cpp(eigenpair, Y, K, min, max, sigma, approach);

    double obj = 0;
    for(auto it=multi_models.begin();it!=multi_models.end();it++) {
      obj += (it->res).obj;
    }

    if(obj>max_obj) {
      max_obj = obj;
      best_multi_models = multi_models;
      best_a2 = a2s[i];
      best_eigenpair = eigenpair_UU;
      best_Z_UU = Z_UU;
    }

  }

  std::cout << "By " << approach << " method, optimal epsilon = " << std::sqrt(best_a2) \
            << ", the objective function is " << max_obj << std::endl;

  std::list<BinaryModel> & multi_models = best_multi_models;

  // test model
  std::cout << "Testing..." << std::endl;
  // Nystrom extension
  Eigen::MatrixXd Z_allU = Eigen::exp(-distances_allU.array()/(best_a2*distances_mean));
  Eigen::MatrixXd A_allU = (1.0/Z_allU.rowwise().sum().array()).matrix().asDiagonal()*Z_allU*(1.0/(best_Z_UU.colwise().sum().array())).matrix().asDiagonal();
  Eigen::MatrixXd W_allU = (1.0/A_allU.rowwise().sum().array()).matrix().asDiagonal()*A_allU;
  EigenPair & eigenpair = best_eigenpair;
  eigenpair.vectors = W_allU*eigenpair.vectors*(1.0/eigenpair.values.array()).matrix().asDiagonal();

  // predict labels on new samples
  Eigen::VectorXd Y_pred = test_logit_mult_gp_cpp(multi_models, eigenpair, m, m_new, K, min, max, sigma);
  std::cout << "Over" << std::endl;

  return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred);
}



Rcpp::List fit_gl_logit_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                int K, Rcpp::NumericVector N_train,
                                double sigma, std::vector<double> a2s,
                                double threshold, bool sparse,
                                std::string approach,
                                Rcpp::List models,
                                bool output_cov) {
  std::cout << "Binary classification with graph Laplacian Gaussian process:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));
  const Eigen::VectorXd N(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(N_train));

  int m = X.rows(); int m_new = X_new.rows(); int n = m + m_new;
  int d = X.cols();

  Eigen::MatrixXd X_all(n,d);
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;

  const std::string subsample = Rcpp::as<std::string>(models["subsample"]);

  /*
  const Eigen::MatrixXd distances  = ((-2.0*X_all*X_all.transpose()).colwise() + X_all.rowwise().squaredNorm()).rowwise() + X_all.rowwise().squaredNorm().transpose();
  double distances_mean = distances.array().sum()/(n*n);
  int r = std::max((int)std::round(threshold*n),3); // r will be greater than 3.
  Rcpp::List res_knn = KNN_cpp(X_all, X_all, r, "Euclidean", true);
  const Eigen::SparseMatrix<double, Eigen::RowMajor>& distances_sp = res_knn["distances_sp"];
  */

  Eigen::MatrixXd distances;
  double distances_mean;
  Eigen::SparseMatrix<double, Eigen::RowMajor> distances_sp;

  if(sparse) {
    int r = std::max((int)std::round(threshold*n),3); // r will be greater than 3.
    Rcpp::List res_knn = KNN_cpp(X_all, X_all, r, "Euclidean", true);
    distances_sp = res_knn["distances_sp"];
    distances_mean = distances_sp.coeffs().sum()/(n*r) * r;
  } else {
    distances = ((-2.0*X_all*X_all.transpose()).colwise() + X_all.rowwise().squaredNorm()).rowwise() + X_all.rowwise().squaredNorm().transpose();
    distances_mean = distances.array().mean();
  }



  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m,0,m-1);

  // train model
  std::cout << "Training..." << std::endl;
  // grid search
  double best_t = 0, best_a2 = 0;
  double max_obj = -std::numeric_limits<double>::infinity();
  EigenPair best_eigenpair;
  int l = a2s.size();

  Rcpp::Environment RSpectra = Rcpp::Environment::namespace_env("RSpectra");
  Rcpp::Function eigs_sym = RSpectra["eigs_sym"];

  for(int i=0;i<l;i++) {
    double a2 = a2s[i];
    EigenPair eigenpair;
    if(sparse) {
      Eigen::SparseMatrix<double> Z(distances_sp);
      Z.coeffs() = Eigen::exp(-Z.coeffs()/(a2*distances_mean));
      Z = (Z + Eigen::SparseMatrix<double>(Z.transpose()))/2.0;
      Eigen::VectorXd Z_rowsum = Z*Eigen::VectorXd::Ones(n);
      Eigen::SparseMatrix<double> A = (1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal()*Z*(1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal();
      Eigen::VectorXd sqrt_D_inv = 1.0/((A*Eigen::VectorXd::Ones(n)).array()+1e-5).sqrt();
      Eigen::SparseMatrix<double> W = sqrt_D_inv.asDiagonal()*A*sqrt_D_inv.asDiagonal();

      Rcpp::List res_eigs = eigs_sym(Rcpp::Named("A")=W,
                                     Rcpp::Named("k")=K);
      eigenpair = EigenPair(Rcpp::as<Eigen::VectorXd>(res_eigs["values"]),
                            Rcpp::as<Eigen::MatrixXd>(res_eigs["vectors"]));
      eigenpair.vectors = std::sqrt(n)*sqrt_D_inv.asDiagonal()*eigenpair.vectors;
    } else {
      Eigen::MatrixXd Z = Eigen::exp(-distances.array()/(a2*distances_mean));

      Eigen::VectorXd Z_rowsum = Z*Eigen::VectorXd::Ones(n);
      Eigen::MatrixXd A = (1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal()*Z*(1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal();
      Eigen::VectorXd sqrt_D_inv = 1.0/((A*Eigen::VectorXd::Ones(n)).array()+1e-5).sqrt();
      Eigen::MatrixXd W = sqrt_D_inv.asDiagonal()*A*sqrt_D_inv.asDiagonal();

      Rcpp::List res_eigs = eigs_sym(Rcpp::Named("A")=W,
                                     Rcpp::Named("k")=K);
      eigenpair = EigenPair(Rcpp::as<Eigen::VectorXd>(res_eigs["values"]),
                            Rcpp::as<Eigen::MatrixXd>(res_eigs["vectors"]));
      eigenpair.vectors = std::sqrt(n)*sqrt_D_inv.asDiagonal()*eigenpair.vectors;
    }

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

    if(res.obj>max_obj) {
      max_obj = res.obj;
      best_t = res.t;
      best_a2 = a2s[i];
      best_eigenpair = eigenpair;
    }

  }

  EigenPair & eigenpair = best_eigenpair;

  std::cout << "By " << approach << " method, optimal epsilon = " << std::sqrt(best_a2) << ", t = " << best_t \
            << ", the objective function is " << max_obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;
  // construct covariance matrix
  Eigen::VectorXi idx0 = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  Eigen::VectorXi idx1 = Eigen::VectorXi::LinSpaced(m_new, m, n-1);
  Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, best_t, idx0, idx0);
  Cvv.diagonal().array() += sigma;
  Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, best_t, idx1, idx0);

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




Rcpp::List fit_gl_logit_mult_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                               int K,
                               double sigma, std::vector<double> a2s,
                               double threshold, bool sparse,
                               std::string approach,
                               Rcpp::List models) {
  std::cout << "Multinomial classification with graph Laplacian Gaussian process:" << std::endl;

  // map the matrices from R to Eigen
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_train));
  const Eigen::VectorXd Y(Rcpp::as<Eigen::Map<Eigen::VectorXd>>(Y_train));
  const Eigen::Map<Eigen::MatrixXd> X_new(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(X_test));

  int m = X.rows(); int m_new = X_new.rows(); int n = m + m_new;
  int d = X.cols();

  int min = Y.minCoeff();
  int max = Y.maxCoeff();

  Eigen::MatrixXd X_all(n,d);
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;

  const std::string subsample = Rcpp::as<std::string>(models["subsample"]);

  /*
  const Eigen::MatrixXd distances  = ((-2.0*X_all*X_all.transpose()).colwise() + X_all.rowwise().squaredNorm()).rowwise() + X_all.rowwise().squaredNorm().transpose();
  double distances_mean = distances.array().sum()/(n*n);
  int r = std::max((int)std::round(threshold*n),3); // r will be greater than 3.
  Rcpp::List res_knn = KNN_cpp(X_all, X_all, r, "Euclidean", true);
  const Eigen::SparseMatrix<double, Eigen::RowMajor>& distances_sp = res_knn["distances_sp"];
  */

  Eigen::MatrixXd distances;
  double distances_mean;
  Eigen::SparseMatrix<double, Eigen::RowMajor> distances_sp;

  if(sparse) {
    int r = std::max((int)std::round(threshold*n),3); // r will be greater than 3.
    Rcpp::List res_knn = KNN_cpp(X_all, X_all, r, "Euclidean", true);
    distances_sp = res_knn["distances_sp"];
    distances_mean = distances_sp.coeffs().sum()/(n*r) * r;
  } else {
    distances = ((-2.0*X_all*X_all.transpose()).colwise() + X_all.rowwise().squaredNorm()).rowwise() + X_all.rowwise().squaredNorm().transpose();
    distances_mean = distances.array().mean();
  }

  Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(m,0,m-1);

  // train model
  std::cout << "Training..." << std::endl;
  // grid search
  double best_a2 = 0;
  std::list<BinaryModel> best_multi_models;
  double max_obj = -std::numeric_limits<double>::infinity();
  EigenPair best_eigenpair;
  int l = a2s.size();

  Rcpp::Environment RSpectra = Rcpp::Environment::namespace_env("RSpectra");
  Rcpp::Function eigs_sym = RSpectra["eigs_sym"];

  for(int i=0;i<l;i++) {
    double a2 = a2s[i];
    EigenPair eigenpair;
    if(sparse) {
      Eigen::SparseMatrix<double> Z(distances_sp);
      Z.coeffs() = Eigen::exp(-Z.coeffs()/(a2*distances_mean));
      Z = (Z + Eigen::SparseMatrix<double>(Z.transpose()))/2.0;
      Eigen::VectorXd Z_rowsum = Z*Eigen::VectorXd::Ones(n);
      Eigen::SparseMatrix<double> A = (1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal()*Z*(1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal();
      Eigen::VectorXd sqrt_D_inv = 1.0/((A*Eigen::VectorXd::Ones(n)).array()+1e-5).sqrt();
      Eigen::SparseMatrix<double> W = sqrt_D_inv.asDiagonal()*A*sqrt_D_inv.asDiagonal();

      Rcpp::List res_eigs = eigs_sym(Rcpp::Named("A")=W,
                                     Rcpp::Named("k")=K);
      eigenpair = EigenPair(Rcpp::as<Eigen::VectorXd>(res_eigs["values"]),
                            Rcpp::as<Eigen::MatrixXd>(res_eigs["vectors"]));
      eigenpair.vectors = std::sqrt(n)*sqrt_D_inv.asDiagonal()*eigenpair.vectors;
    } else {
      Eigen::MatrixXd Z = Eigen::exp(-distances.array()/(a2*distances_mean));

      Eigen::VectorXd Z_rowsum = Z*Eigen::VectorXd::Ones(n);
      Eigen::MatrixXd A = (1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal()*Z*(1.0/(Z_rowsum.array()+1e-5)).matrix().asDiagonal();
      Eigen::VectorXd sqrt_D_inv = 1.0/((A*Eigen::VectorXd::Ones(n)).array()+1e-5).sqrt();
      Eigen::MatrixXd W = sqrt_D_inv.asDiagonal()*A*sqrt_D_inv.asDiagonal();

      Rcpp::List res_eigs = eigs_sym(Rcpp::Named("A")=W,
                                     Rcpp::Named("k")=K);
      eigenpair = EigenPair(Rcpp::as<Eigen::VectorXd>(res_eigs["values"]),
                            Rcpp::as<Eigen::MatrixXd>(res_eigs["vectors"]));
      eigenpair.vectors = std::sqrt(n)*sqrt_D_inv.asDiagonal()*eigenpair.vectors;
    }

    // empirical Bayes to optimize t
    std::list<BinaryModel> multi_models = train_logit_mult_gp_cpp(eigenpair, Y, K, min, max, sigma, approach);

    double obj = 0;
    for(auto it=multi_models.begin();it!=multi_models.end();it++) {
      obj += (it->res).obj;
    }

    if(obj>max_obj) {
      max_obj = obj;
      best_multi_models = multi_models;
      best_a2 = a2s[i];
      best_eigenpair = eigenpair;
    }

  }

  EigenPair & eigenpair = best_eigenpair;
  std::list<BinaryModel> & multi_models = best_multi_models;

  std::cout << "By " << approach << " method, optimal epsilon = " << std::sqrt(best_a2) \
            << ", the objective function is " << max_obj << std::endl;

  // test model
  std::cout << "Testing..." << std::endl;

  // predict labels on new samples
  Eigen::VectorXd Y_pred = test_logit_mult_gp_cpp(multi_models, eigenpair, m, m_new, K, min, max, sigma);
  std::cout << "Over" << std::endl;

  return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred);

}
