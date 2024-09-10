// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>

#include <iostream>
#include <exception>
#include <cmath>

#include "Utils.h"




// inverse logit link function
double ilogit(double x) {
  return 1/(1+exp(-x));
}

// Stick breaking transform from f to pi
Eigen::MatrixXd f_to_pi(const Eigen::MatrixXd & f) {
  Eigen::MatrixXd pi = 1.0/(1.0+Eigen::exp(-f.array()));
  return pi;
}

// Predict Y based on pi
Eigen::VectorXd pi_to_Y(const Eigen::VectorXd & pi) {
  return (pi.array()>0.5).cast<double>();
}


Eigen::MatrixXd subsample_cpp(const Eigen::MatrixXd & X, int s, std::string method, int nstart) {
  int n = X.rows(); int d = X.cols();
  Eigen::MatrixXd U;

  if(method=="kmeans") {
    Rcpp::Environment stats = Rcpp::Environment::namespace_env("stats");
    Rcpp::Function kmeans = stats["kmeans"];
    Rcpp::List cluster_kmeans = kmeans(Rcpp::Named("x")=Rcpp::wrap(X),
                                       Rcpp::Named("centers")=s,
                                       Rcpp::Named("iter.max")=100,
                                       Rcpp::Named("nstart")=nstart);
    U.resize(s, d+1);
    U.leftCols(d) = Rcpp::as<Eigen::MatrixXd>(cluster_kmeans["centers"]);
    U.col(d) =  Rcpp::as<Eigen::VectorXd>(cluster_kmeans["size"]);
  } else if(method=="random") {
    Eigen::VectorXi rows = Rcpp::as<Eigen::VectorXi>(Rcpp::sample(n, s)).array()-1;
    U = mat_indexing(X, rows, Eigen::VectorXi::LinSpaced(d,0,d-1));
  } else if(method=="minibatchkmeans") {
    Rcpp::Environment ClusterR = Rcpp::Environment::namespace_env("ClusterR");
    Rcpp::Function MiniBatchKmeans = ClusterR["MiniBatchKmeans"];
    Rcpp::List mbkmeans = MiniBatchKmeans(Rcpp::Named("data")=Rcpp::wrap(X),
                                          Rcpp::Named("clusters")=s,
                                          Rcpp::Named("batch_size")=s*10,
                                          Rcpp::Named("init_fraction")=s*20.0/n,
                                          Rcpp::Named("num_init")=nstart);
    U.resize(s, d+1);
    U.leftCols(d) = Rcpp::as<Eigen::MatrixXd>(mbkmeans["centroids"]);
    Eigen::VectorXi labels = KNN_cpp(X, U.leftCols(d), 1)["ind_knn"];
    for(int i=0; i<s;i++) {
      U(i,d) = (labels.array()==i).count();
    }
  } else {
    Rcpp::stop("The subsample method is not supported!");
  }

  return U;
}



struct KNN_Index : public RcppParallel::Worker {
  // source matrix
  const Eigen::MatrixXd & input;

  // destination matrix
  Eigen::MatrixXi & output;

  int r;
  int s;

  // initialize from Rcpp input and output matrices (the RMatrix class
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
                   std::string distance, bool output, int batch) {
  int n = X.rows();
  int s = U.rows();


  Eigen::MatrixXi distances_ind(n,r);
  // Eigen::SparseMatrix<double, Eigen::RowMajor> distances_sp(n,s);
  Eigen::MatrixXd distances_mat;
  Eigen::MatrixXi distances_ind_batch;

  int K = std::floor(n/batch);
  int R = n % batch;

  // whether to compute the sparse distance matrix
  if(!output) {
    for(int i=0;i<K;i++) {
      const Eigen::MatrixXd & X_batch = X.middleRows(i*batch, batch);
      if(distance=="Euclidean") {
        distances_mat = ((-2*X_batch*U.transpose()).colwise() + X_batch.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
      } else {
        Rcpp::stop("The distance method of KNN is not supported!\n");
      }

      distances_ind_batch.resize(batch,r);
      KNN_Index knn_index(distances_mat, distances_ind_batch, r);
      RcppParallel::parallelFor(0, batch, knn_index);
      distances_ind.middleRows(i*batch,batch) = distances_ind_batch;
    }

    const Eigen::MatrixXd & X_batch = X.bottomRows(R);
    if(distance=="Euclidean") {
      distances_mat = ((-2*X_batch*U.transpose()).colwise() + X_batch.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
    } else {
      Rcpp::stop("The distance method of KNN is not supported!\n");
    }

    distances_ind_batch.resize(R,r);
    KNN_Index knn_index(distances_mat, distances_ind_batch, r);
    RcppParallel::parallelFor(0, R, knn_index);
    distances_ind.bottomRows(R) = distances_ind_batch;

    return Rcpp::List::create(Rcpp::Named("ind_knn")=distances_ind);
  } else{
    Eigen::SparseMatrix<double, Eigen::RowMajor> distances_sp(n,s);
    distances_sp.reserve(Eigen::VectorXi::Constant(n,r));

    for(int i=0;i<K;i++) {
      const Eigen::MatrixXd & X_batch = X.middleRows(i*batch, batch);
      if(distance=="Euclidean") {
        distances_mat = ((-2*X_batch*U.transpose()).colwise() + X_batch.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
      } else {
        Rcpp::stop("The distance method of KNN is not supported!\n");
      }

      distances_ind_batch.resize(batch,r);
      KNN_Index knn_index(distances_mat, distances_ind_batch, r);
      RcppParallel::parallelFor(0, batch, knn_index);
      distances_ind.middleRows(i*batch,batch) = distances_ind_batch;

      for(int j=0;j<batch;j++) {
        for(int k=0;k<r;k++) {
          int indk = distances_ind_batch(j,k);
          distances_sp.insert(i*batch+j,indk) = distances_mat(j,indk);
        }
      }
    }

    const Eigen::MatrixXd & X_batch = X.bottomRows(R);
    if(distance=="Euclidean") {
      distances_mat = ((-2*X_batch*U.transpose()).colwise() + X_batch.rowwise().squaredNorm()).rowwise() + U.rowwise().squaredNorm().transpose();
    } else {
      Rcpp::stop("The distance method of KNN is not supported!\n");
    }

    distances_ind_batch.resize(R,r);
    KNN_Index knn_index(distances_mat, distances_ind_batch, r);
    RcppParallel::parallelFor(0, R, knn_index);
    distances_ind.bottomRows(R) = distances_ind_batch;

    for(int j=0;j<R;j++) {
      for(int k=0;k<r;k++) {
        int indk = distances_ind_batch(j,k);
        distances_sp.insert(n-R+j,indk) = distances_mat(j,indk);
      }
    }

    return Rcpp::List::create(Rcpp::Named("ind_knn")=distances_ind, Rcpp::Named("distances_sp")=distances_sp);
  }

}


void graphLaplacian_cpp(Eigen::SparseMatrix<double,Eigen::RowMajor>& Z,
                        std::string gl,
                        const Eigen::VectorXd & num_class) {
  if(gl=="rw") {}
  else if(gl=="normalized") {
    Eigen::VectorXd Z_colsum = Eigen::RowVectorXd::Ones(Z.rows()) * Z;
    Z = Z * (1.0/(Z_colsum.array()+1e-9)).matrix().asDiagonal();
  } else if(gl=="cluster-normalized") {
    Eigen::VectorXd Z_colsum = Eigen::RowVectorXd::Ones(Z.rows()) * Z;
    Z = Z * (1.0/(Z_colsum.array()+1e-9)).matrix().asDiagonal();
    Z = Z * num_class.asDiagonal();
  } else {
    Rcpp::stop("Error: the type of graph Laplacian is not supported!");
  }

  Eigen::VectorXd Z_rowsum = Z * Eigen::VectorXd::Ones(Z.cols());
  Z = (1.0/(Z_rowsum.array()+1e-9)).matrix().asDiagonal() * Z;
}


Eigen::VectorXd posterior_covariance_regression(const EigenPair & eigenpair,
                                     const Eigen::VectorXi & idx0, const Eigen::VectorXi & idx1,
                                     int K, const std::vector<double> & pars, double sigma) {
  int m = idx0.rows();
  double t = pars[0];
  double var = pars[1];
  Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(K).array();
  const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
  Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(K,0,K-1);
  Eigen::MatrixXd V2 = mat_indexing(eigenvectors, idx1, cols);
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Lambda = (Eigen::exp(-t*eigenvalues.array())).matrix().asDiagonal();
  Eigen::VectorXd beta;

  if(m<=K) {
    Eigen::MatrixXd C11 = HK_from_spectrum_cpp(eigenpair, K, t, idx0, idx0);
    Eigen::MatrixXd K11 = C11;
    K11.diagonal().array() += var + sigma;
    Eigen::MatrixXd C21 = HK_from_spectrum_cpp(eigenpair, K, t, idx1, idx0);

    Eigen::LLT<Eigen::MatrixXd> chol_K(K11);
    Eigen::MatrixXd alpha = C21*chol_K.solve(Eigen::MatrixXd::Identity(m,m));
    beta = (C21.array()*alpha.array()).rowwise().sum();
  } else {
    Eigen::MatrixXd V1 = mat_indexing(eigenvectors, idx0, cols);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Lambda_sqrt = (Eigen::exp(-0.5*t*eigenvalues.array())+0.0).matrix().asDiagonal();
    Eigen::MatrixXd Q = Lambda_sqrt*V1.transpose()*V1*Lambda_sqrt;
    Q.diagonal().array() += var + sigma;
    Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
    Eigen::MatrixXd alpha = 1.0/(var+sigma)*Lambda*V1.transpose()*(V1 - V1*Lambda_sqrt*chol_Q.solve(Lambda_sqrt*(V1.transpose()*V1)))*Lambda;
    beta = (V2.array()*(V2*alpha).array()).rowwise().sum();
  }

  Eigen::VectorXd cov = ((V2*Lambda).array()*V2.array()).rowwise().sum() + var + sigma - beta.array();
  return cov;
}


Rcpp::List posterior_distribution_classification(const Eigen::MatrixXd & C11, const Eigen::MatrixXd & C21, const Eigen::VectorXd & C22,
                                                 const Eigen::VectorXd & Y,
                                                 double tol, int max_iter) {
  int m = Y.rows();

  // initialize f
  Eigen::VectorXd f = Eigen::VectorXd::Constant(m, 0);

  Eigen::VectorXd pi, W, b, a, f_new;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> sqrt_W;
  Eigen::MatrixXd B;
  Eigen::LLT<Eigen::MatrixXd> chol_B;


  // Newton method
  // locate posterior mode by Algorithm 3.1 in GPML
  for(int iter=0;iter<max_iter;iter++) {
    pi = f_to_pi(f);
    W = pi.array()*(1-pi.array());
    sqrt_W = W.array().sqrt().matrix().asDiagonal();
    B = sqrt_W*C11*sqrt_W;
    B.diagonal().array() += 1;
    chol_B = B.llt();
    b = W.array()*f.array() + (Y.array()-pi.array());
    a = b - sqrt_W*chol_B.solve(sqrt_W*(C11*b));
    f_new = C11*a;

    if((f-f_new).lpNorm<1>()<tol) {
      f = f_new;
      break;
    } else {
      f = f_new;
    }
  }

  // compute posterior mean and covariance by Algorithm 3.2 in GPML
  pi = f_to_pi(f);
  W = pi.array()*(1-pi.array());
  sqrt_W = W.array().sqrt().matrix().asDiagonal();
  B = sqrt_W*C11*sqrt_W;
  B.diagonal().array() += 1;
  chol_B = B.llt();

  Eigen::VectorXd mean = C21*(Y-pi);
  Eigen::VectorXd beta = sqrt_W*chol_B.solve(Eigen::MatrixXd::Identity(m,m))*sqrt_W;
  Eigen::VectorXd cov = C22.array() - ((C21*beta).array()*C21.array()).rowwise().sum();
  return Rcpp::List::create(Rcpp::Named("mean")=mean, Rcpp::Named("cov")=cov);
}


double negative_log_likelihood(const Eigen::MatrixXd & mean, const Eigen::MatrixXd & cov, const Eigen::VectorXd & target, std::string type) {
  double nll = 0;

  if(type=="regression") {
    nll = (((target-mean).array().square()/cov.array() + (cov.array()+1e-9).log()).mean() + std::log(2*3.1415926))/2;
  } else if(type=="binary") {
    nll = nll_classification(mean, cov, target);
  } else if(type=="multinomial") {
    // split test data
    Eigen::MatrixXd aug_y = multi_train_split(target);
    for(int j=0;j<aug_y.cols();j++) {
      nll += nll_classification(mean.col(j), cov.col(j), aug_y.col(j));
    }
  }

  return nll;
}


double nll_classification(const Eigen::VectorXd & mean, const Eigen::VectorXd & cov, const Eigen::VectorXd & target, int n_samples) {
  int n = mean.rows();
  // random sampling
  Eigen::MatrixXd rand_mat = Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(Rcpp::rnorm(n*n_samples));
  rand_mat.resize(n,n_samples);
  Eigen::MatrixXd f_samples = rand_mat.array().colwise()*cov.array().sqrt();
  f_samples.colwise() += mean;

  // MC integral
  Eigen::MatrixXd pi_samples = f_to_pi(f_samples);
  Eigen::MatrixXd like_samples = pi_samples.array().colwise()*target.array() + (1.0-pi_samples.array()).colwise()*(1.0-target.array());
  Eigen::VectorXd like = like_samples.array().rowwise().mean();
  double nll = -(like.array()+1e-2).log().mean();

  return nll;
}


Rcpp::List posterior_distribution_multiclassification(const EigenPair & eigenpair, const MultiClassifier & multiclassifier,
                                                      const Eigen::VectorXi & idx, const Eigen::VectorXi & idx_new,
                                                      int K, double sigma) {
  const Eigen::MatrixXd & aug_y = multiclassifier.aug_y;
  const std::vector<ReturnValue> & res_vec = multiclassifier.res_vec;
  int J = aug_y.cols();

  Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(K).array();
  const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
  Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(K,0,K-1);
  Eigen::MatrixXd V2 = mat_indexing(eigenvectors, idx_new, cols);

  int m_new = idx_new.rows();
  Eigen::MatrixXd mean(m_new,J);
  Eigen::MatrixXd cov(m_new,J);

  for(int j=0;j<J;j++) {
    const ReturnValue & res = res_vec[j];
    Eigen::MatrixXd C11 = HK_from_spectrum_cpp(eigenpair, K, res.t, idx, idx);
    Eigen::MatrixXd C21 = HK_from_spectrum_cpp(eigenpair, K, res.t, idx_new, idx);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Lambda = (Eigen::exp(-res.t*eigenvalues.array())).matrix().asDiagonal();
    Eigen::VectorXd C22 = ((V2*Lambda).array()*V2.array()).rowwise().sum() + sigma;

    Rcpp::List post = posterior_distribution_classification(C11, C21, C22, aug_y.col(j));
    mean.col(j) = Rcpp::as<Eigen::VectorXd>(post["mean"]);
    cov.col(j) = Rcpp::as<Eigen::VectorXd>(post["cov"]);
  }

  return Rcpp::List::create(Rcpp::Named("mean")=mean,
                            Rcpp::Named("cov")=cov);
}





