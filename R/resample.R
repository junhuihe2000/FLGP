#' Resample omega given N and f
#'
#' @param aug_data An augmented data for polya-gamma sampling.
#'
#' @return `omega` A numeric matrix with dim(m, J-1), the updated omega.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' aug_data = AugmentedData(C,Y)
#' resample_omega(aug_data)
resample_omega <- function(aug_data=AugmentedData()) {
  m = aug_data$m; J = aug_data$J
  omega = matrix(0, m, J-1)
  # Update auxiliary variables according to their conditional Poyla-Gamma distributions
  for(i in c(1:m)) {
    for(j in c(1:(J-1))) {
      omega[i,j] = BayesLogit::rpg(1,aug_data$N[i,j],aug_data$f[i,j])
    }
  }
  return(omega)
}


#' Resample f given omega and y
#'
#' @param aug_data An augmented data for polya-gamma sampling.
#'
#' @return `f` A numeric matrix with dim(m,J-1), the updated f.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' aug_data = AugmentedData(C,Y)
#' resample_f(aug_data)
resample_f <- function(aug_data=AugmentedData()) {
  m = aug_data$m; J = aug_data$J
  f = matrix(0, m, J-1)
  C_inv = aug_data$C_inv

  for(j in c(1:(J-1))) {
    # compute posterior parameters
    C_post = C_inv + diag(aug_data$omega[,j])
    R_post = chol(C_post)
    mu_post = solve(C_post, aug_data$kappa[,j])

    # sample f from posterior Gaussian distributions
    rand_vec = stats::rnorm(m)
    f[,j] = mu_post + backsolve(R_post, rand_vec)
  }
  return(f)
}
