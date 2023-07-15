#' Robust marginal log likelihood in Gaussian regression
#'
#' @param eigenpair A list includes values and vectors.
#' @param Y A numeric vector with length(m), response vector.
#' @param theta A numeric vector, hyper-parameters.
#' @param K An integer, the number of used eigenpairs.
#'
#' @return `mll` A double, the marginal log likelihood value.
#' @export
#'
#' @examples
#' X <- matrix(rnorm(3*3), 3, 3)
#' Y <- rowSums(X)
#' X_new <- matrix(rnorm(10*3),10,3)
#' s <- 5; r <- 3
#' m <- 3; K <- 3
#' eigenpair <- heat_kernel_spectrum(X, X_new, s, r, K=K)
#' theta <- c(1,0.1)
#' K <- 3
#' marginal_log_likelihood_reg_robust(eigenpair, Y, theta, K)
marginal_log_likelihood_reg_robust <-function(eigenpair, Y, theta, K) {
  # using matrix identities equation A.9 and A.10 in GPML

  m = length(Y)
  vectors = eigenpair$vectors[1:m,1:K]
  values = eigenpair$values

  if(FALSE) {
  Q = Matrix::crossprod(vectors[1:m,]) + (theta[2]^2)*diag(exp((1-values)*theta[1])) + 1e-9*diag(1,K)
  a = Matrix::t(vectors[1:m,])%*%Y

  mll = -0.5*(sum(Y^2)-sum(a*Matrix::solve(Q, a)))/(theta[2]^2+1e-9)
  logdet = Matrix::determinant(exp(-theta[1]/K*sum(1-values))*Q, logarithm = TRUE)$modulus[1]
  mll = mll - 0.5*(logdet + (m-K)*log(theta[2]^2+1e-9))
  }

  Umk = Matrix::colScale(vectors[1:m,], exp(-0.5*theta[1]*(1-values)))
  Ukm = Matrix::t(Umk)
  Q = Ukm%*%Umk + (theta[2]^2+1e-9)*diag(1,K)
  a = Ukm%*%Y

  mll = -0.5*(sum(Y^2)-sum(a*Matrix::solve(Q,a)))/(theta[2]^2+1e-9)
  logdet = Matrix::determinant(Q, logarithm = TRUE)$modulus[1]
  mll = mll - 0.5*(logdet + (m-K)*log(theta[2]^2+1e-9))

  return(mll)
}
