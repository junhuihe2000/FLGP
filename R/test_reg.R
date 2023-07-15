#' Predict response on new samples for Gaussian regression
#'
#' @param C A numeric matrix with dim(m,m), the noisy self covariance matrix
#' in the training samples.
#' @param Y A numeric vector with length(m), response vector.
#' @param Cnv A numeric matrix with dim(m_new,m), cross covariance matrix
#' between new sample and training sample.
#'
#' @return `Y_pred` A numeric vector with length(m_new), predictive response vector.
#' @export
#'
#' @examples
#' Z <- matrix(rnorm(3*3),3,3)
#' C <- Z%*%t(Z)
#' Y <- rnorm(3)
#' Cnv <- matrix(rnorm(5*3),5,3)
#' test_pgbinary(C, Y, Cnv)
test_reg <- function(C, Y, Cnv) {
  m = length(Y)
  C[cbind(rep(1:m),rep(1,m))] = C[cbind(rep(1:m),rep(1,m))] + 1e-3
  # posterior expectation
  Y_pred = Cnv%*%solve(C, Y)
  return(as.vector(Y_pred))
}
