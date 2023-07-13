#' Marginal log likelihood by stick-breaking given C and Y
#'
#' @param C A numeric matrix with dim(m,m), the self covariance matrix
#' in the training samples.
#' @param Y A numeric matrix with dim(m,J), each row is a one hot vector indicating
#' the label.
#' @param N_sample An integer, the length of the Gibbs sampler chain.
#'
#' @return `mll` A double, the marginal log likelihood.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' marginal_log_likelihood(C, Y, 100)
marginal_log_likelihood <- function(C, Y, N_sample=100) {
  aug_data = AugmentedData(C, Y)
  aug_data = PG_Gibbs_sampler(aug_data, N_sample)
  mll = log_likelihood(aug_data$f, Y)
  return(mll)
}
