#' Learn diffusion time t by maximizing log marginal likelihood
#'
#' @param eigenpair A list includes values and vectors.
#' @param Y A numeric vector with length(m), count of the positive class.
#' @param m An integer, the number of labeled samples.
#' @param K An integer, the number of used eigenpairs.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param N A numeric vector with length(m), total count.
#' @param t0 A positive double, the initial guess for t, defaulting value `10`.
#' @param lower the lower bound on t, defaulting value `0`.
#' @param upper the upper bound on t, defaulting value `100`.
#'
#' @return A list with two components
#' \describe{
#' \item{t}{the optimal diffusion time.}
#' \item{mll}{the corresponding optimal log marginal likelihood value.}
#' }
#' @export
#'
#' @examples
#' X0 <- matrix(rnorm(3*3), 3, 3)
#' X1 <- matrix(rnorm(3*3, 5), 3, 3)
#' Y <- c(1,1,1,0,0,0)
#' X <- rbind(X0,X1)
#' X0_new <- matrix(rnorm(10*3),10,3)
#' X1_new <- matrix(rnorm(10*3, 2),10,3)
#' X_new <- rbind(X0_new, X1_new)
#' s <- 6; r <- 3
#' m <- 6; K <- 3
#' eigenpair <- heat_kernel_spectrum(X, X_new, s, r, K=K)
#' train_lae_logit_gp(eigenpair, Y, m, K)
train_lae_logit_gp <- function(eigenpair, Y, m, K, sigma=1e-3, N=NULL, t0=NULL, lower=0, upper=100) {
  eigenpair$vectors = eigenpair$vectors[1:m,]
  # initialize t
  if(is.null(t0)) {
    t0 = 10
  }

  # optimize negative log marginal likelihood
  opt = stats::optim(t0,
                     negative_marginal_likelihood_logit(eigenpair, Y, m, K, N, sigma),
                     NULL,
                     method = "L-BFGS-B",
                     lower = lower,
                     upper = upper)
  t = opt$par
  mll = -opt$value
  cat("For local anchor embedding, optimal t =",t,", log marginal likelihood is",mll,".\n")
  return(list(t=t,mll=mll))
}
