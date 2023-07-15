#' Create negative log posterior functional with inverse gamma prior for logistic regression
#'
#' @param eigenpair A list includes values and vectors.
#' @param Y A numeric vector with length(m), count of the positive class.
#' @param idx An integer vector with length(m), the index of training samples.
#' @param K An integer, the number of used eigenpairs.
#' @param N A numeric vector with length(m), total count.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param p A positive double, hyper-parameter, defaulting value is 1e-2.
#' @param q A positive double, hyper-parameter, defaulting value is 10.
#' @param tau A positive double, hyper-parameter, defaulting value is 2.
#'
#' @return `nlp` negative log posterior functional with parameter t.
#' @export
#'
#' @examples
#' Z <- matrix(rnorm(3*3),3,3)
#' X <- Z%*%t(Z)
#' eigenpair <- eigen(X)
#' Y <- sample(c(0,1),3, replace=TRUE)
#' negative_log_posterior_logit(eigenpair, Y, c(1:3), 2)
negative_log_posterior_logit <- function(eigenpair, Y, idx, K, N=NULL, sigma=1e-3,
                                         p=1e-2, q=10, tau=2) {
  m = length(idx)
  nll = negative_marginal_likelihood_logit(eigenpair, Y, idx, K, N=NULL, sigma=1e-3)

  nlp <- function(t) {
    nlpr = p*log(t+1e-5) + (t/tau)^{-q}
    return(nll(t)+nlpr)
  }

  return(nlp)
}
