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
Eigen::VectorXd f_to_pi(const Eigen::VectorXd & f) {
  Eigen::VectorXd pi = 1/(1+Eigen::exp(-f.array()));
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



