#' Graph Laplacian type
#'
#' @param Z A numeric matrix with dim(n,s), similarity matrix.
#' @param gl A character vector, the type of GL.
#' @param num_class A numeric vector, the number of points in each cluster
#'
#' @return `Z` A numeric matrix with dim(n,s), normalized Z.
#' @export
#'
#' @examples
#' Z <- matrix(runif(5*2),5,2)
#' gl <- "rw"
#' graphLaplacian(Z, gl)
graphLaplacian <- function(Z, gl, num_class=NULL) {
  if(gl=="rw") {
    Z = Matrix::rowScale(Z, Matrix::rowSums(Z)^{-1})
  } else if(gl=="normalized") {
    Z_norm = Matrix::colScale(Z, Matrix::colSums(Z)^{-1})
    Z = Matrix::rowScale(Z_norm, Matrix::rowSums(Z_norm)^{-1})
  } else if(gl=="cluster-normalized") {
    Z_norm = Matrix::colScale(Z, Matrix::colSums(Z)^{-1})
    Z_cl = Matrix::colScale(Z_norm, num_class)
    Z = Matrix::rowScale(Z_cl, Matrix::rowSums(Z_cl)^{-1})
  } else {
    stop("Error: the type of graph Laplacian is not supported!")
  }
  return(Z)
}
