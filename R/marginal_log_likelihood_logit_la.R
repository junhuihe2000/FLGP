#' Mode-finding for binary Laplace GPC with logit link function
#'
#' @param C A numeric matrix with dim(m,m), covariance matrix.
#' @param Y A numeric vector with length(m), count of the positive class.
#' @param N A numeric vector with length(m), total count.
#' @param f A numeric vector with length(m), initial guess for newton methods,
#' the defaulting value is `NULL`, that is, `f=rep(0,m)`.
#' @param tol A double, convergence criterion, the defaulting value is `1e-5`.
#' @param max_iter An integer, the maximum iteration number, defaulting value `100`.
#'
#' @return A list with three components.
#' \describe{
#' \item{f}{the posterior mode.}
#' \item{amll}{the Laplace approximation of the marginal log likelihood.}
#' \item{converge}{0 indicates Newton method succeeds to converge,
#' 1 indicates Newton method fails to converge.}
#' }
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- sample(c(0,1), 3,replace=TRUE)
#' marginal_log_likelihood_logit_la(C, Y)
marginal_log_likelihood_logit_la <- function(C, Y, N=NULL, f=NULL, tol=1e-5, max_iter=100) {
  stopifnot(is.vector(Y))
  m = nrow(C)
  if(is.null(f)) {
    f = rep(0,m)
  }
  if(is.null(N)) {
    N = rep(1,m)
  }

  # Newton method
  # Algorithm 3.1 in GPML
  iter = 0
  converge = 1
  while(iter<max_iter){
    iter = iter + 1

    pi = ilogit(f)
    W = N*pi*(1-pi)
    sqrtW = sqrt(W)
    B = diag(1,m) + Matrix::dimScale(C, sqrtW)
    R = chol(B)
    b = W*f + Y*(1-pi) + (N-Y)*(-pi)
    a = b - sqrtW*backsolve(R, forwardsolve(t(R), sqrtW*(C%*%cbind(b))))
    f_new = C%*%cbind(a)

    if(sum(abs(f-f_new))<tol) {
      f = f_new
      converge = 0
      break
    } else {
      f = f_new
    }
  }

  pi = ilogit(f)
  # approximate marginal log likelihood
  amll = -0.5*sum(a*f)
  amll = amll + sum(Y*log(pi)) + sum((N-Y)*log(1-pi))
  amll = amll - sum(log(diag(R)))
  return(list(f=f,amll=amll,converge=converge))
}
