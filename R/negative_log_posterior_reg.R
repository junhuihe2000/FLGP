#' Create negative log posterior functional with inverse gamma prior for regression
#'
#' @param eigenpair A list includes values and vectors.
#' @param Y A numeric vector with length(m), response vector.
#' @param K An integer, the number of used eigenpairs.
#' @param p A positive double, hyper-parameter, defaulting value is 1e-2.
#' @param q A positive double, hyper-parameter, defaulting value is 10.
#' @param tau A positive double, hyper-parameter, defaulting value is 2.
#'
#' @return `nlp` negative log posterior functional with parameter theta,
#' where theta = c(t, sigma).
#' @export
#'
#' @examples
#' Z <- matrix(rnorm(3*3),3,3)
#' X <- Z%*%t(Z)
#' eigenpair <- eigen(X)
#' Y <- rnorm(3)
#' K <- 2
#' negative_log_posterior_reg(eigenpair, Y, K)
negative_log_posterior_reg <- function(eigenpair, Y, K,
                                             p=1e-2, q=10, tau=2) {
  nll = negative_marginal_likelihood_reg(eigenpair, Y, K)
  nlp <- function(theta) {
    nlpr = nlpr = p*log(theta[1]+1e-5) + (theta[1]/tau)^{-q}
    return(nll(theta)+nlpr)
  }
  return(nlp)
}
