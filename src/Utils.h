#ifndef UTILS_H
#define UTILS_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

using namespace Rcpp;
using namespace Eigen;


//' k-nearest neighbor reference points
//'
//' @param X Original points, a (n,d) matrix, each row indicates one original point.
//' @param U Reference points, a (s,d) or (s,d+1) matrix, each row indicates one reference point.
//' @param r The number of k-nearest neighbor points, an integer.
//' @param distance The distance method to compute k-nearest neighbor points, characters in c("Euclidean", "geodesic"),
//'  including Euclidean distance and geodesic distance, the defaulting distance
//'  is Euclidean distance.
//' @param output Bool, whether to output the distance matrix, defaulting value is `FALSE`.
//'
//' @return If `output=FALSE`, `list(ind_knn)`, the indexes of KNN, a list with length n, each component of the list is a vector of length r,
//'  indicating the indexes of KNN for the corresponding original point based on the chosen distance.
//'  Otherwise `output=TRUE`, `list(ind_knn,distances_sp)`, a list with two components, the one is the indexes of KNN,
//'  the other is the sparse distance matrix with dim(n,s).
//' @export
//'
//' @examples
//' X <- matrix(rnorm(300), nrow=100, ncol=3)
//' U <- matrix(rnorm(30), nrow=10, ncol=3)
//' r <- 3
//' distance <- "Euclidean"
//' KNN_cpp(X, U, r, distance)
// [[Rcpp::export(KNN_cpp)]]
Rcpp::List KNN_cpp(const Eigen::MatrixXd & X, const Eigen::MatrixXd & U, int r = 3,
                  Rcpp::String distance = "Euclidean", bool output = false);



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
