#' Construct stick breaking multinomial Gaussian process models
#'
#' @param C A numeric matrix with dim(m, m), the covariance matrix of training samples,
#' where `C[i,j]` indicates the covariance between `X_i` and `X_j`.
#' @param Y A numeric matrix with dim(m,J), each row is a one hot vector indicating
#' the label.
#'
#' @return An instance of the class `AugmentedData`.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' AugmentedData(C,Y)
AugmentedData <- function(C, Y) {
  stopifnot(methods::is(C, "matrix")||methods::is(C, "Matrix"),
            methods::is(Y,  "matrix")||methods::is(C, "Matrix"))
  stopifnot(nrow(C)==ncol(C), nrow(C)==nrow(Y))
  J = ncol(Y); m = nrow(Y)
  # compute C_inv firstly to reuse
  C_inv = solve(C)
  N = N_vec(Y); kappa = kappa_vec(Y, N)

  # initialize the auxiliary variable
  omega = matrix(1, m, J-1)

  # initialize a "sample" from f
  f = matrix(0, m, J-1)

  aug_data = list(J=J,
               m=m,
               C=C,
               C_inv=C_inv,
               Y=Y,
               N=N,
               kappa=kappa,
               omega=omega,
               f=f)
  class(aug_data) = "AugmentedData"
  return(aug_data)
}
