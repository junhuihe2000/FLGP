#' The spectrum of the similarity matrix W from Z
#' @description The transition matrix W is the two step composition of
#' the cross similarity matrix Z by $W=ZLambda^{-1}Z^T$.
#'
#' @param Z A numeric matrix with dim (n,s), the cross similarity matrix
#' between the original sample and the sub-sample.
#' @param K An integer, the number of eigenpairs requested, the defaulting value
#' is NULL, indicating all non-trivial eigenpairs, that is, K=min(n,s).
#' @param root A logical value, indicating whether to square root eigenvalues of W,
#' the defaulting value is FALSE.
#'
#' @return A list of converged eigenvalues and eigenvectors of W.
#' \describe{
#' \item{values}{eigenvalues, descending order.}
#' \item{vectors}{eigenvectors, the vectors are normalized to sqrt(n) length.}
#' }
#' @export
#'
#' @examples
#' Z <- matrix(abs(rnorm(5*3)),5,3)
#' Z <- Matrix::rowScale(Z, Matrix::rowSums(Z)^{-1})
#' spectrum_from_Z(Z)
spectrum_from_Z <- function(Z, K=NULL, root=FALSE) {
  stopifnot(methods::is(Z, "matrix")||methods::is(Z, "Matrix"))
  A = Matrix::colScale(Z, sqrt(Matrix::colSums(Z)+1e-5)^{-1})
  eigenpairs = truncated_SVD(A, K)
  if(root) {
    eigenpairs$values = sqrt(eigenpairs$values)
  }
  n = nrow(Z)
  eigenpairs$vectors = sqrt(n)*eigenpairs$vectors
  return(eigenpairs)
}
