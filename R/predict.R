#' Compute the log likelihood of the observed
#'
#' @param f A numeric matrix with dim(M,J-1), latent function.
#' @param Y A numeric matrix with dim(m,J), each row is a one hot vector indicating
#' the label.
#'
#' @return `ll` a double, log likelihood value.
#' @export
#'
#' @examples
#' f <- matrix(rnorm(3*2),3,2)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' log_likelihood(f, Y)
log_likelihood <- function(f, Y) {
  stopifnot(nrow(f)==nrow(Y), ncol(Y)-ncol(f)==1)

  pi = f_to_pi(f)

  ll = sum(lgamma(rowSums(Y)+1)) - sum(lgamma(Y+1)) + sum(Y*log(pi))
  return(ll)
}


# Predict the GP value at the inputs X_new by Cnv and evaluate the log likelihood of Y_new
predictive_log_likelihood <- function(aug_data=AugmentedData(), Cnv, Y_new) {
  pi = collapsed_predict(aug_data, Cnv)
  pll = sum(lgamma(rowSums(Y_new)+1)) - sum(lgamma(Y_new+1)) + sum(Y_new*log(pi))
  return(pll)
}


#' Predict multinomial probability given omega
#' @description  Predict the multinomial probability vector at Cnv
#' by first integrating out the value of f, given
#' omega and the kernel parameters.
#'
#' @param aug_data An augmented data for polya-gamma sampling.
#' @param Cnv A numeric matrix with dim(m_new,m), cross covariance matrix
#' between new sample and training sample.
#'
#' @return `pis_new` a numeric matrix with dim(m_new, J), each row indicates
#' the predictive multinomial probability in the corresponding new sample point.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' aug_data <- AugmentedData(C,Y)
#' Cnv <- matrix(rnorm(5*3),5,3)
#' collapsed_predict(aug_data, Cnv)
collapsed_predict <- function(aug_data=AugmentedData(), Cnv) {
  stopifnot(methods::is(aug_data, "AugmentedData"), ncol(Cnv)==aug_data$m)
  m_new = nrow(Cnv)

  mu_fs_new = matrix(0, m_new, aug_data$J-1)

  for(j in c(1:(aug_data$J-1))) {
    omegaj = aug_data$omega[,j] + 1e-16
    kappaj = aug_data$kappa[,j]
    # account for the mean from the omega potentials
    y = kappaj / omegaj
    Cvv_noisy = aug_data$C + diag(1/omegaj)
    mu_fs_new[,j] = Cnv%*%solve(Cvv_noisy, y)
  }

  # convert f to pi
  pis_new = f_to_pi(mu_fs_new)

  return(pis_new)
}


#' Predict labels given omega and Cnv
#'
#' @param aug_data An augmented data for polya-gamma sampling.
#' @param Cnv A numeric matrix with dim(m_new,m), cross covariance matrix
#' between new sample and training sample.
#'
#' @return `Y_new` A numeric matrix with dim(m_new, J), each row indicates
#' the predictive "label" of the corresponding new sample.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' aug_data <- AugmentedData(C,Y)
#' Cnv <- matrix(rnorm(5*3), 5, 3)
#' collapsed_predict_label(aug_data, Cnv)
collapsed_predict_label <- function(aug_data=AugmentedData(), Cnv) {
  pis_new = collapsed_predict(aug_data, Cnv)
  m_new = nrow(Cnv); J = aug_data$J
  Y_new = matrix(0, m_new, J)
  # find the maximum index of pi
  ind_new = apply(pis_new, 1, which.max)

  Y_new[cbind(c(1:m_new),ind_new)] = 1L

  return(Y_new)
}
