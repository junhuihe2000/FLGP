#' Predict multinomial probabilities on new samples
#'
#' @param C A numeric matrix with dim(m,m), the self covariance matrix
#' in the training samples.
#' @param Y A numeric matrix with dim(m,J), each row is a one hot vector indicating
#' the label.
#' @param Cnv A numeric matrix with dim(m_new,m), cross covariance matrix
#' between new sample and training sample.
#' @param N_sample An integer, the length of the Gibbs sampler chain.
#'
#' @return `pis_new` A numeric matrix with dim(m_new,J), each row indicates
#' the predictive multinomial probability in the corresponding new sample point.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' Cnv <- matrix(rnorm(5*3),5,3)
#' test_pgmult(C, Y, Cnv, 100)
test_pgmult <- function(C, Y, Cnv, N_sample=100) {
  stopifnot(nrow(C)==nrow(Y), nrow(C)==ncol(Cnv))
  aug_data = AugmentedData(C, Y)
  aug_data = PG_Gibbs_sampler(aug_data, N_sample)
  pis_new = collapsed_predict(aug_data, Cnv)
  return(pis_new)
}


#' Predict labels on new samples
#'
#' @param C A numeric matrix with dim(m,m), the self covariance matrix
#' in the training samples
#' @param Y A numeric matrix with dim(m,J), each row is a one hot vector indicating
#' the label.
#' @param Cnv A numeric matrix with dim(m_new,m), cross covariance matrix
#' between new sample and training sample.
#' @param N_sample An integer, the length of the Gibbs sampler chain.
#'
#' @return `Y_new` A numeric matrix with dim(m_new,J), each row indicates
#' the label in the corresponding new sample point.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' Cnv <- matrix(rnorm(5*3),5,3)
#' test_pgmult_label(C, Y, Cnv, 100)
test_pgmult_label <- function(C, Y, Cnv, N_sample=1000) {
  stopifnot(nrow(C)==nrow(Y), nrow(C)==ncol(Cnv))
  aug_data = AugmentedData(C, Y)
  aug_data = PG_Gibbs_sampler(aug_data, N_sample)
  Y_new = collapsed_predict_label(aug_data, Cnv)
  return(Y_new)
}
