#' Marginal log likelihood in Gaussian regression
#'
#' @param C_noisy A numeric matrix with dim(m,m), noisy covariance matrix,
#' `C_noisy=C+sigma^2I` .
#' @param Y A numeric vector with length(m), response vector.
#'
#' @return `mll` A double, the marginal log likelihood value.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' C_noisy <- C + diag(1e-3, 3)
#' Y <- rnorm(3)
#' marginal_log_likelihood_logit_la(C_noisy, Y)
marginal_log_likelihood_reg <- function(C_noisy, Y) {
  m = nrow(C_noisy)
  C_noisy[cbind(rep(1:m),rep(1:m))] = C_noisy[cbind(rep(1:m),rep(1:m))] + 1e-3
  # using equation 5.8 in GPML
  mll = -0.5*sum(Y*solve(C_noisy, Y))
  mll = mll - 0.5*log(Matrix::det(C_noisy))
  return(mll)
}
