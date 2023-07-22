#' Construct stick breaking multinomial Gaussian process models
#'
#' @param C A numeric matrix with dim(m, m), the covariance matrix of training samples,
#' where `C[i,j]` indicates the covariance between `X_i` and `X_j`.
#' @param Y A numeric matrix with dim(m,J), each row is a one hot vector indicating
#' the label.
#' @param inverse Bool, whether to compute C inverse firstly, defaulting value is FALSE.
#'
#' @return An instance of the class `AugmentedData`.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' AugmentedData(C,Y)
AugmentedData <- function(C, Y, inverse=FALSE) {
  stopifnot(methods::is(C, "matrix")||methods::is(C, "Matrix"),
            methods::is(Y,  "matrix")||methods::is(Y, "Matrix"))
  stopifnot(nrow(C)==ncol(C), nrow(C)==nrow(Y))
  J = ncol(Y); m = nrow(Y)

  N = N_vec(Y); kappa = kappa_vec(Y, N)

  # initialize the auxiliary variable
  omega = matrix(1, m, J-1)

  # initialize a "sample" from f
  f = matrix(0, m, J-1)

  aug_data = list(J=J,
               m=m,
               C=C,
               Y=Y,
               N=N,
               kappa=kappa,
               omega=omega,
               f=f)

  # compute C_inv firstly to reuse
  if(inverse) {
    C_inv = solve(C)
    aug_data$C_inv = C_inv
  }

  class(aug_data) = "AugmentedData"
  return(aug_data)
}
