#' Learn diffusion time t by maximizing log marginal likelihood or log posterior
#'
#' @param eigenpair A list includes values and vectors.
#' @param Y A numeric vector with length(m), count of the positive class.
#' @param idx An integer vector with length(m), the index of training samples.
#' @param K An integer, the number of used eigenpairs.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param N A numeric vector with length(m), total count.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param t0 A positive double, the initial guess for t, defaulting value `10`.
#' @param lower the lower bound on t, defaulting value `1e-3`.
#' @param upper the upper bound on t, defaulting value `100`.
#'
#' @return A list with two components
#' \describe{
#' \item{t}{the optimal diffusion time.}
#' \item{obj}{the corresponding optimal objective function value.}
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
#' train_lae_logit_gp(eigenpair, Y, c(1:m), K)
train_lae_logit_gp <- function(eigenpair, Y, idx, K, sigma=1e-3, N=NULL,
                               approach="posterior", t0=NULL, lower=1e-3, upper=100) {
  m = length(idx)
  eigenpair$vectors = eigenpair$vectors[idx,]
  # initialize t
  if(is.null(t0)) {
    t0 = 10
  }

  if(approach=="marginal") {
    # optimize negative log marginal likelihood
    opt = stats::optim(t0,
                       negative_marginal_likelihood_logit(eigenpair, Y, c(1:m), K, N, sigma),
                       NULL,
                       method = "L-BFGS-B",
                       lower = lower,
                       upper = upper)
    cat("For local anchor embedding, optimal t =",opt$par,", log marginal likelihood is",-opt$value,".\n")
  } else if(approach=="posterior") {
    # optimize negative log posterior
    opt = stats::optim(t0,
                       negative_log_posterior_logit(eigenpair, Y, c(1:m), K, N, sigma),
                       NULL,
                       method = "L-BFGS-B",
                       lower = lower,
                       upper = upper)
    cat("For local anchor embedding, optimal t =",opt$par,", log posterior is",-opt$value,".\n")
  } else {
    stop("This model selection approach is not supported!")
  }

  t = opt$par
  obj = -opt$value
  return(list(t=t,obj=obj))
}
