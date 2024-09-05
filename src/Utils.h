#ifndef UTILS_H
#define UTILS_H


// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <list>

#include "Spectrum.h"
#include "MultiClassification.h"


Eigen::MatrixXd f_to_pi(const Eigen::MatrixXd & f);
double ilogit(double x);
Eigen::VectorXd pi_to_Y(const Eigen::VectorXd & pi);


//' Subsample in the domain
//'
//' @param X Original sample, a (n, d) matrix, each row indicates one point in R^d.
//' @param s An integer indicating the number of the subsampling.
//' @param method How to subsample, characters in c("kmeans", "random"),
//' including kmeans and random selection, the
//' defaulting subsampling method is kmeans.
//' @param nstart An integer, the number of random sets chosen in kmeans,
//' the defaulting value is `1`.
//'
//' @return A subsampling, a (s, d) or (s, d+1) matrix, each row indicates one point in R^d,
//' where the d+1 column indicates the number of points in each cluster if it exists.
//' @export
//'
//' @examples
//' X <- matrix(rnorm(10*3), nrow=10, ncol=3)
//' s <- 3
//' U = subsample_cpp(X, s, method = "kmeans")
// [[Rcpp::export(subsample_cpp)]]
Eigen::MatrixXd subsample_cpp(const Eigen::MatrixXd & X, int s, std::string method = "kmeans",
                              int nstart = 1);

// Graph Laplacian type
void graphLaplacian_cpp(Eigen::SparseMatrix<double,Eigen::RowMajor>& Z,
                        std::string gl = "rw",
                        const Eigen::VectorXd & num_class = Eigen::VectorXd());


//' k-nearest neighbor reference points
//'
//' @param X Original points, a (n,d) matrix, each row indicates one original point.
//' @param U Reference points, a (s,d) or (s,d+1) matrix, each row indicates one reference point.
//' @param r The number of k-nearest neighbor points, an integer.
//' @param distance The distance method to compute k-nearest neighbor points, characters in c("Euclidean", "geodesic"),
//'  including Euclidean distance and geodesic distance, the defaulting distance
//'  is Euclidean distance.
//' @param output Bool, whether to output the distance matrix, defaulting value is `FALSE`.
//' @param batch Int, the batch size, defaulting value is `100`.
//'
//' @returns If `output=FALSE`, `list(ind_knn)`, the indexes of KNN, a list with length n, each component of the list is a vector of length r,
//'  indicating the indexes of KNN for the corresponding original point based on the chosen distance.
//'  Otherwise `output=TRUE`, `list(ind_knn,distances_sp)`, a list with two components, the one is the indexes of KNN,
//'  the other is the sparse distance matrix with dim(n,s).
// [[Rcpp::export(KNN_cpp)]]
Rcpp::List KNN_cpp(const Eigen::MatrixXd & X, const Eigen::MatrixXd & U, int r = 3,
                  std::string distance = "Euclidean", bool output = false, int batch=100);

// compute the posterior covariance of test data in GPR
Eigen::VectorXd posterior_covariance_regression(const EigenPair & eigenpair,
                                     const Eigen::VectorXi & idx0, const Eigen::VectorXi & idx1,
                                     int K, const std::vector<double> & pars, double sigma);

//' compute the posterior distribution of test data in GPC
//' @param C11 Prior covariance matrix on training data.
//' @param C21 Prior covariance matrix between test data and training data.
//' @param C22 Prior covariance matrix on test data.
//' @param Y Training labels.
//'
//' @export
//'
// [[Rcpp::export(posterior_distribution_classification)]]
Rcpp::List posterior_distribution_classification(const Eigen::MatrixXd & C11, const Eigen::MatrixXd & C21, const Eigen::VectorXd & C22,
                                                 const Eigen::VectorXd & Y,
                                                 double tol=1e-5, int max_iter=100);

// compute the posterior distribution of test data in GPC
Rcpp::List posterior_distribution_multiclassification(const EigenPair & eigenpair, const std::list<BinaryModel> & models,
                                                      int m, int m_new, int K, double sigma);

//' compute the negative log likelihood of test data
//'
//' @param mean Posterior mean.
//' @param cov Posterior covariance.
//' @param target Targeted labels.
//' @param type The task type, `regression`, `binary` or `multinomial`.
//'
//' @return negative log likelihood.
//' @export
//'
// [[Rcpp::export(negative_log_likelihood)]]
double negative_log_likelihood(const Eigen::MatrixXd & mean, const Eigen::MatrixXd & cov, const Eigen::VectorXd & target, std::string type);

// compute the negative log likelihood of test data in binary classification
double nll_classification(const Eigen::VectorXd & mean, const Eigen::VectorXd & cov, const Eigen::VectorXd & target, int n_samples=100);

/*
 * Matrix indexing rows and columns
*/
template<class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor {
  const ArgType &m_arg;
  const RowIndexType &m_rowIndices;
  const ColIndexType &m_colIndices;
public:
  typedef Eigen::Matrix<typename ArgType::Scalar,
                        RowIndexType::SizeAtCompileTime,
                        ColIndexType::SizeAtCompileTime,
                        ArgType::Flags&Eigen::RowMajorBit?Eigen::RowMajor:Eigen::ColMajor,
                        RowIndexType::MaxSizeAtCompileTime,
                        ColIndexType::MaxSizeAtCompileTime> MatrixType;

  indexing_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
    : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
  {}

  const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
    return m_arg(m_rowIndices[row], m_colIndices[col]);
  }
};


// indexing rows and columns
template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<indexing_functor<ArgType,RowIndexType,ColIndexType>, typename indexing_functor<ArgType,RowIndexType,ColIndexType>::MatrixType>
mat_indexing(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
{
  typedef indexing_functor<ArgType,RowIndexType,ColIndexType> Func;
  typedef typename Func::MatrixType MatrixType;
  return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}


#endif
