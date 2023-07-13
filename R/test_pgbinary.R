#' Predict labels on new samples with Polya-Gamma
#'
#' @param C A numeric matrix with dim(m,m), the self covariance matrix
#' in the training samples.
#' @param Y A numeric vector with length(m), count of the positive class.
#' @param Cnv A numeric matrix with dim(m_new,m), cross covariance matrix
#' between new sample and training sample.
#' @param N A numeric vector with length(m), total count.
#' @param N_sample An integer, the length of the Gibbs sampler chain.
#' @param full_out Bool, whether or not to output pi_new, defaulting value is `FALSE`.
#'
#' @return `Y_pred` if `full_out=FALSE`, otherwise `list(pi_pred,Y_pred)`.
#' @export
#'
#' @examples
#' Z <- matrix(rnorm(3*3),3,3)
#' C <- Z%*%t(Z)
#' Y <- sample(c(0,1),3, replace=TRUE)
#' Cnv <- matrix(rnorm(5*3),5,3)
#' test_pgbinary(C, Y, Cnv)
test_pgbinary <- function(C, Y, Cnv, N=NULL, N_sample=1000, full_out=FALSE) {
  m = nrow(C)
  if(is.null(N)) {
    N = rep(1,m)
  }

  # predict pi by Polya-Gamma auxiliary variables
  pi_pred = test_pgmult(C, cbind(Y,N-Y), Cnv, N_sample)[,1]
  Y_pred = vapply(pi_pred, function(pi) {
    if(pi>0.5) {return(1)}
    else {return(0)}
  }, FUN.VALUE = 0)

  if(full_out) {
    return(list(pi_pred=pi_pred, Y_pred=Y_pred))
  }

  return(Y_pred)
}
