// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>

#include <iostream>
#include <exception>

#include "Utils.h"


/*
using namespace Rcpp;
using namespace Eigen;
*/


// inverse logit link function
double ilogit(double x) {
  return 1/(1+exp(-x));
}

// Stick breaking transform from f to pi
Eigen::VectorXd f_to_pi(const Eigen::VectorXd & f) {
  Eigen::VectorXd pi = 1/(1+Eigen::exp(-f.array()));
  return pi;
}

// Predict Y based on pi
Eigen::VectorXd pi_to_Y(const Eigen::VectorXd & pi) {
  return (pi.array()>0.5).cast<double>();
}


Eigen::MatrixXd subsample_cpp(const Eigen::MatrixXd & X, int s, std::string method) {
  int n = X.rows(); int d = X.cols();
  Eigen::MatrixXd U;

  if(method=="kmeans") {
    Rcpp::Environment stats = Rcpp::Environment::namespace_env("stats");
    Rcpp::Function kmeans = stats["kmeans"];
    Rcpp::List cluster_kmeans = kmeans(Rcpp::Named("x")=Rcpp::wrap(X),
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



struct KNN_Index : public RcppParallel::Worker {
  // source matrix
  const Eigen::MatrixXd & input;

  // destination matrix
  Eigen::MatrixXi & output;

  int r;
  int s;

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
                   std::string distance, bool output) {
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

  /*
  for(int i=0;i<n;i++) {
    Eigen::ArrayXi ind = Eigen::ArrayXi::LinSpaced(s, 0, s-1);
    const Eigen::RowVectorXd & row = distances_mat.row(i);
    std::partial_sort(ind.data(), ind.data()+r, ind.data()+s, [&row](int i1, int i2) {return row(i1)<row(i2);});
    distances_ind.row(i) = ind.head(r);
  }
  */

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


void graphLaplacian_cpp(Eigen::SparseMatrix<double,Eigen::RowMajor>& Z,
                        std::string gl,
                        const Eigen::VectorXd & num_class) {
  if(gl=="rw") {}
  else if(gl=="normalized") {
    Eigen::VectorXd Z_colsum = Eigen::RowVectorXd::Ones(Z.rows()) * Z;
    Z = Z * (1.0/Z_colsum.array()).matrix().asDiagonal();
  } else if(gl=="cluster-normalized") {
    Eigen::VectorXd Z_colsum = Eigen::RowVectorXd::Ones(Z.rows()) * Z;
    Z = Z * (1.0/Z_colsum.array()).matrix().asDiagonal();
    Z = Z * num_class.asDiagonal();
  } else {
    Rcpp::stop("Error: the type of graph Laplacian is not supported!");
  }

  Eigen::VectorXd Z_rowsum = Z * Eigen::VectorXd::Ones(Z.cols());
  Z = (1.0/Z_rowsum.array()).matrix().asDiagonal() * Z;
}


/*
Eigen::MatrixXd mini_batch_kmeans(Eigen::MatrixXd& data, int clusters, int batch_size, int max_iters, int num_init,
                                  double init_fraction, std::string initializer,
                                  int early_stop_iter, bool verbose,
                                  Rcpp::Nullable<Rcpp::NumericMatrix> CENTROIDS,
                                  double tol, double tol_optimal_init, int seed) {
  Rcpp::Environment ClusterR = Rcpp::Environment::namespace_env("ClusterR");
  Rcpp::Function MiniBatchKmeans = ClusterR["MiniBatchKmeans"];
  Rcpp::List res = MiniBatchKmeans(Rcpp::Named("data")=Rcpp::wrap(data),
                                   Rcpp::Named("clusters")=clusters, Rcpp::Named("batch_size")=batch_size,
                                   Rcpp::Named("num_init")=num_init, Rcpp::Named("max_iters")=max_iters,
                                   Rcpp::Named("init_fraction")=init_fraction, Rcpp::Named("initializer")=initializer,
                                   Rcpp::Named("early_stop_iter")=early_stop_iter, Rcpp::Named("verbose")=verbose,
                                   Rcpp::Named("CENTROIDS")=CENTROIDS, Rcpp::Named("tol")=tol,
                                   Rcpp::Named("tol_optimal_init")=tol_optimal_init, Rcpp::Named("seed")=seed);
  Eigen::MatrixXd centroids = Rcpp::as<Eigen::MatrixXd>(res["centroids"]);
  return centroids;
}



Eigen::VectorXd Predict_mini_batch_kmeans(Eigen::MatrixXd& data, Eigen::MatrixXd& CENTROIDS,
                                       bool fuzzy, bool updated_output) {
  Rcpp::Environment ClusterR = Rcpp::Environment::namespace_env("ClusterR");
  Rcpp::Function predict_MBatchKMeans = ClusterR["predict_MBatchKMeans"];
  Rcpp::NumericVector res = predict_MBatchKMeans(Rcpp::Named("data")=Rcpp::wrap(data), Rcpp::Named("CENTROIDS")=Rcpp::wrap(CENTROIDS),
                                        Rcpp::Named("fuzzy")=fuzzy, Rcpp::Named("updated_output")=updated_output);
  Eigen::VectorXd clusters = Rcpp::as<Eigen::VectorXd>(res);

  return clusters;
}
*/


