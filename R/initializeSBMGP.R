#' Compute the count vector for PG Multinomial inference
#'
#' @param Y A numeric matrix with dim(m,J), each row is a one hot vector indicating
#' the label.
#'
#' @return `N` a numeric matrix with dim(m,J-1), each row indicates the count vector.
#' @export
#'
#' @examples
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' N_vec(Y)
N_vec <- function(Y) {
  stopifnot(ncol(Y)>=2)
  M = nrow(Y); J = ncol(Y)
  N = matrix(0, M, J-1)
  N[,1] = rowSums(Y)
  if(J>2) {
    for(j in c(2:(J-1))) {
      N[,j] = N[,j-1] - Y[,j-1]
    }
  }
  if(sum(abs(N[,J-1]-Y[,J-1]-Y[,J]))>.Machine$double.eps^0.5) {
    stop("Error: N fails to compute!")
  }
  return(N)
}


#' Compute the kappa vector for PG Multinomial inference
#'
#' @param Y A numeric matrix with dim(m,J), each row is a one hot vector indicating
#' the label.
#' @param N A numeric matrix with dim(m,J-1), each row indicates the count vector.
#'
#' @return `kappa` a numeric matrix with dim(m, J-1).
#' @export
#'
#' @examples
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' kappa_vec(Y, N_vec(Y))
kappa_vec <- function(Y, N) {
  stopifnot(ncol(Y)-ncol(N)==1)
  M = nrow(Y); J = ncol(Y)
  kappa = Y[,-J] - N/2
  return(kappa)
}
