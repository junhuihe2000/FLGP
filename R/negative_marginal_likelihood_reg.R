#' Create negative log marginal likelihood functional for regression
#'
#' @param eigenpair A list includes values and vectors.
#' @param Y A numeric vector with length(m), response vector.
#' @param K An integer, the number of used eigenpairs.
#'
#' @return `nll` negative log marginal likelihood functional with parameter theta,
#' where theta = c(t, sigma).
#' @export
#'
#' @examples
#' Z <- matrix(rnorm(3*3),3,3)
#' X <- Z%*%t(Z)
#' eigenpair <- eigen(X)
#' Y <- rnorm(3)
#' K <- 2
#' negative_marginal_likelihood_reg(eigenpair, Y, K)
negative_marginal_likelihood_reg <- function(eigenpair, Y, K) {
  m = length(Y)
  idx = c(1:m)
  nll <- function(theta) {
    if(FALSE) {
    C = HK_from_spectrum(eigenpair, K, theta[1], idx, idx)
    C[cbind(rep(1,m),rep(1,m))] = C[cbind(rep(1,m),rep(1,m))] + theta[2]^2
    mll = marginal_log_likelihood_reg(C, Y)
    }

    mll = marginal_log_likelihood_reg_robust(eigenpair, Y, theta, K)
    return(-mll)
  }
  return(nll)
}
