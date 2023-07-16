#' Learn hyper-parameters by maximizing log marginal likelihood or log posterior
#'
#' @param eigenpair A list includes values and vectors.
#' @param Y A numeric vector with length(m), response vector.
#' @param K An integer, the number of used eigenpairs.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param theta0 The initial guess for theta, defaulting value `c(10,1)`.
#' @param lower the lower bound on theta, defaulting value `c(1e-3,-Inf)`.
#' @param upper the upper bound on theta, defaulting value `c(Inf,Inf)`.
#'
#' @return A list with two components
#' \describe{
#' \item{theta}{the optimal hyper-parameters.}
#' \item{obj}{the corresponding optimal objective function value.}
#' }
#' @export
#'
#' @examples
#' X <- matrix(rnorm(3*3), 3, 3)
#' Y <- rowSums(X)
#' X_new <- matrix(rnorm(10*3),10,3)
#' s <- 5; r <- 3
#' m <- 3; K <- 3
#' eigenpair <- heat_kernel_spectrum(X, X_new, s, r, K=K)
#' train_lae_reg_gp(eigenpair, Y, K)
train_lae_reg_gp <- function(eigenpair, Y, K, approach="posterior",
                             theta0=NULL, lower = c(1e-3, -Inf), upper = c(Inf, Inf)) {
  m = length(Y)
  eigenpair$vectors = eigenpair$vectors[1:m,]
  # initialize theta
  if(is.null(theta0)) {
    theta0 = c(10, 1)
  }

  if(approach=="marginal") {
    # optimize negative log marginal likelihood
    opt = stats::optim(theta0,
                       negative_marginal_likelihood_reg(eigenpair, Y, K),
                       NULL,
                       method = "L-BFGS-B",
                       lower = lower,
                       upper = upper)
    cat("For local anchor embedding, optimal t =",opt$par[1],", sigma2 =",
        opt$par[2]^2,", log marginal likelihood is",-opt$value,".\n")
  } else if(approach=="posterior") {
    # optimize negative log posterior
    opt = stats::optim(theta0,
                       negative_log_posterior_reg(eigenpair, Y, K),
                       NULL,
                       method = "L-BFGS-B",
                       lower = lower,
                       upper = upper)
    cat("For local anchor embedding, optimal t =",opt$par[1],", sigma2 =",
        opt$par[2]^2,", log posterior is",-opt$value,".\n")
  } else {
    stop("This model selection approach is not supported!")
  }

  theta = opt$par
  obj = -opt$value
  return(list(theta=theta,obj=obj))
}
