#' Train Gaussian process logistic multinomial regression with local anchor embedding kernels
#'
#' @description Compose J-1 binary logistic regression to implement the multinomial regression.
#'
#' @param eigenpair A list includes values and vectors.
#' @param Y A numeric vector with length(m), each element indicates the label,
#' taking value in `c(0:(J-1))`.
#' @param K An integer, the number of used eigenpairs.
#' @param J An integer, the number of classes.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param N A numeric vector with length(m), total count.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param t0 A positive double, the initial guess for t, defaulting value `10`.
#' @param lower the lower bound on t, defaulting value `1e-3`.
#' @param upper the upper bound on t, defaulting value `100`.
#'
#' @return A model list with J-1 model, each model includes four components
#' \describe{
#' \item{Y}{the re-encoded training samples.}
#' \item{idx}{the index of Y in original samples.}
#' \item{t}{the optimal diffusion time.}
#' \item{obj}{the corresponding optimal objective function value.}
#' }
#' @export
#'
#' @examples
#' X0 <- matrix(rnorm(3*3), 3, 3)
#' X1 <- matrix(rnorm(3*3, 5), 3, 3)
#' X2 <- matrix(rnorm(3*3, -5), 3, 3)
#' Y <- c(0,0,0,1,1,1,2,2,2)
#' X <- rbind(X0,X1,X2)
#' X0_new <- matrix(rnorm(10*3),10,3)
#' X1_new <- matrix(rnorm(10*3, 5),10,3)
#' X2_new <- matrix(rnorm(10*3, -5),10,3)
#' X_new <- rbind(X0_new, X1_new, X2_new)
#' s <- 6; r <- 3
#' K <- 3
#' J <- 3
#' eigenpair <- heat_kernel_spectrum(X, X_new, s, r, K=K)
#' train_lae_logit_mult_gp(eigenpair, Y, K, J)
train_lae_logit_mult_gp <- function(eigenpair, Y, K, J, sigma=1e-3, N=NULL,
                                    approach="posterior", t0=NULL, lower=1e-3, upper=100) {
  m = length(Y)
  eigenpair$vectors = eigenpair$vectors[1:m,]

  model_list = list()

  for(j in c(0:(J-2))) {
    model = list()
    idx = which(Y>=j)
    # encode Yj for the j-th binary logit GPC
    Yj = Y[idx]
    idx1 = which(Yj==j); idx0 = which(Yj>j)
    Yj[idx1] = 1; Yj[idx0] = 0
    # train the j-th binary logit GPC
    res = train_lae_logit_gp(eigenpair, Yj, idx, K, sigma, N, approach, t0, lower, upper)
    model$Y = Yj; model$idx = idx
    model$t = res$t; model$obj = res$obj
    model_list[[j+1]] = model
  }
  return(model_list)
}
