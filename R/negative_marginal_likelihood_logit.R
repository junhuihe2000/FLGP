#' Create negative log marginal likelihood functional for logistic regression
#'
#' @param eigenpair A list includes values and vectors.
#' @param Y A numeric vector with length(m), count of the positive class.
#' @param idx An integer vector with length(m), the index of training samples.
#' @param K An integer, the number of used eigenpairs.
#' @param N A numeric vector with length(m), total count.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#'
#' @return `nll` negative log marginal likelihood functional with parameter t.
#' @export
#'
#' @examples
#' Z <- matrix(rnorm(3*3),3,3)
#' X <- Z%*%t(Z)
#' eigenpair <- eigen(X)
#' Y <- sample(c(0,1),3, replace=TRUE)
#' negative_marginal_likelihood_logit(eigenpair, Y, c(1:3), 2)
negative_marginal_likelihood_logit <- function(eigenpair, Y, idx, K, N=NULL, sigma=1e-3) {
  m = length(idx)
  nll <- function(t) {
    C = HK_from_spectrum(eigenpair, K, t, idx, idx)
    C[cbind(rep(1:m),rep(1:m))] = C[cbind(rep(1:m),rep(1:m))] + sigma
    mll = marginal_log_likelihood_logit_la(as.matrix(C), Y, N)$amll
    return(-mll)
  }
  return(nll)
}
