#' Compute the log likelihood of the observed
#'
#' @param f A numeric matrix with dim(M,J-1), latent function.
#' @param Y A numeric matrix with dim(m,J), each row is a one hot vector indicating
#' the label.
#'
#' @return `ll` a double, log likelihood value.
#' @export
log_likelihood <- function(f, Y) {
  stopifnot(all(dim(f)==dim(Y)))

  pi = f_to_pi(f)

  ll = sum(lgamma(rowSums(Y)+1)) - sum(lgamma(Y+1)) + sum(Y*log(pi))
  return(ll)
}


# Predict the GP value at the inputs X_new by Cnv and evaluate the log likelihood of Y_new
predictive_log_likelihood <- function(model=SBMultinomialGP(), Cnv, Y_new) {
  pi = collapsed_predict(model, Cnv)
  pll = sum(lgamma(rowSums(Y_new)+1)) - sum(lgamma(Y_new+1)) + sum(Y_new*log(pi))
  return(pll)
}


#' Predict multinomial probability given omega
#' @description  Predict the multinomial probability vector at Cnv
#' by first integrating out the value of f, given
#' omega and the kernel parameters.
#'
#' @param model A SBMultinomialGP model.
#' @param Cnv A numeric matrix with dim(m_new,m), cross covariance matrix
#' between new sample and training sample.
#'
#' @return `pis_new` a numeric matrix with dim(m_new, J), each row indicates
#' the predictive multinomial probability in the corresponding new sample point.
#' @export
#'
collapsed_predict <- function(model=SBMultinomialGP(), Cnv) {
  stopifnot(methods::is(model, "SBMultinomialGP"), ncol(Cnv)==model$m)
  m_new = nrow(Cnv)

  mu_fs_new = matrix(0, m_new, model$J-1)

  for(j in c(1:(model$J-1))) {
    omegaj = model$omega[,j] + 1e-16
    kappaj = model$kappa[,j]
    # account for the mean from the omega potentials
    y = kappaj / omegaj
    Cvv_noisy = model$C + diag(1/omegaj)
    mu_fs_new[,j] = Cnv%*%solve(Cvv_noisy, y)
  }

  # convert f to pi
  pis_new = f_to_pi(mu_fs_new)

  return(pis_new)
}

