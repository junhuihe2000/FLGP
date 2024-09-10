# R interfaces for Fit.cpp

#################################################################
# Regression
#################################################################




#' Fit Gaussian process regression with local anchor embedding kernels
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), the training labels.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param r An integer, the number of the nearest neighbor points.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param noise A character vector, taking value in c("same", "different"),
#' indicates whether the noise variances in different locations are different,
#' defaulting value is `same`.
#' @param models A list with four components
#' \describe{
#' \item{subsample}{the method of subsampling in c(`kmeans`, `random`, `minibatchkmeans`), the defaulting value is `kmeans`.}
#' \item{kernel}{the type of kernel to compute cross similarity matrix W, the
#' defaulting value is `lae`.}
#' \item{gl}{the kind of graph Laplacian L, the defaulting value is `cluster-normalized`.}
#' \item{root}{whether to square root eigenvalues of the two steps similarity matrix W,
#' the defaulting value is `TRUE`.}
#' }
#' @param output_cov Bool, whether to output covariance, defaulting value is `FALSE`.
#' @param nstart Int, the number of random sets chosen in kmeans.
#'
#' @return `res` A list with three components including `pars`, `posterior` and `Y_pred`,
#' where `Y_pred` is also a list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
#' }
#' @export
#'
#' @examples
#' X <- matrix(runif(6),3,2)
#' Y <- X[,1]^2 + X[,2]^2
#' X_new <- matrix(runif(10),5,2)
#' Y_new <- X_new[,1]^2 + X_new[,2]^2
#' s <- 6; r <- 3
#' K <- 5
#' lae <- fit_lae_regression_gp_rcpp(X, Y, X_new, s, r, K)
fit_lae_regression_gp_rcpp <- function(X, Y, X_new, s, r, K=-1, sigma=1e-5,
                                  approach="posterior", noise="same",
                                  models=list(subsample="kmeans",
                                              kernel="lae",
                                              gl="cluster-normalized",
                                              root=TRUE),
                                  output_cov=FALSE,
                                  nstart=1) {

  res = fit_lae_regression_gp_cpp(X,Y,X_new,s,r,K,sigma,approach,noise,models,output_cov,nstart)

  return(res)

}


#' Fit Gaussian process regression with square exponential kernels
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), the training labels.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param r An integer, the number of the nearest neighbor points.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param a2s A numeric vector, the searching range for bandwidth.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param noise A character vector, taking value in c("same", "different"),
#' indicates whether the noise variances in different locations are different,
#' defaulting value is `same`.
#' @param models A list with four components
#' \describe{
#' \item{subsample}{the method of subsampling in c(`kmeans`, `random`, `minibatchkmeans`), the defaulting value is `kmeans`.}
#' \item{kernel}{the type of kernel to compute cross similarity matrix W, the
#' defaulting value is `lae`.}
#' \item{gl}{the kind of graph Laplacian L, the defaulting value is `cluster-normalized`.}
#' \item{root}{whether to square root eigenvalues of the two steps similarity matrix W,
#' the defaulting value is `TRUE`.}
#' }
#' @param output_cov Bool, whether to output covariance, defaulting value is `FALSE`.
#' @param nstart Int, the number of random sets chosen in kmeans.
#'
#' @return `res` A list with three components including `pars`, `posterior` and `Y_pred`,
#' where `Y_pred` is also a list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
#' }
#' @export
#'
#' @examples
#' X <- matrix(runif(6),3,2)
#' Y <- X[,1]^2 + X[,2]^2
#' X_new <- matrix(runif(10),5,2)
#' Y_new <- X_new[,1]^2 + X_new[,2]^2
#' s <- 6; r <- 3
#' K <- 5
#' se <- fit_se_regression_gp_rcpp(X, Y, X_new, s, r, K)
fit_se_regression_gp_rcpp <- function(X, Y, X_new, s, r, K=-1, sigma=1e-5, a2s=NULL,
                                 approach="posterior", noise="same",
                                 models=list(subsample="kmeans",
                                             kernel="lae",
                                             gl="cluster-normalized",
                                             root=TRUE),
                                 output_cov=FALSE,
                                 nstart=1) {

  if(is.null(a2s)) {
    a2s = exp(seq(log(0.1),log(10),length.out=10))
  }

  res = fit_se_regression_gp_cpp(X,Y,X_new,s,r,K,sigma,a2s,approach,noise,models,output_cov,nstart)

  return(res)

}


#' Fit Gaussian process regression with Nystrom extension
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), the training labels.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param a2s A numeric vector, the searching range for bandwidth.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param noise A character vector, taking value in c("same", "different"),
#' indicates whether the noise variances in different locations are different,
#' defaulting value is `same`.
#' @param subsample The method of subsampling in c(`kmeans`, `random`, `minibatchkmeans`), the defaulting value is `kmeans`.
#' @param output_cov Bool, whether to output covariance, defaulting value is `FALSE`.
#' @param nstart Int, the number of random sets chosen in kmeans.
#'
#' @return `res` A list with three components including `pars`, `posterior` and `Y_pred`,
#' where `Y_pred` is also a list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
#' }
#' @export
#'
#' @examples
#' X <- matrix(runif(6),3,2)
#' Y <- X[,1]^2 + X[,2]^2
#' X_new <- matrix(runif(10),5,2)
#' Y_new <- X_new[,1]^2 + X_new[,2]^2
#' s <- 6
#' K <- 5
#' nystrom <- fit_nystrom_regression_gp_rcpp(X, Y, X_new, s, K)
fit_nystrom_regression_gp_rcpp <- function(X, Y, X_new, s, K=-1, sigma=1e-5, a2s=NULL,
                                      approach="posterior", noise="same",
                                      subsample="kmeans",
                                      output_cov=FALSE,
                                      nstart=1) {

  if(K<0) {
    K = s
  }

  if(is.null(a2s)) {
    a2s = exp(seq(log(0.1),log(10),length.out=10))
  }

  res = fit_nystrom_regression_gp_cpp(X,Y,X_new,s,K,sigma,a2s,approach,noise,subsample,output_cov,nstart)

  return(res)

}



#' Fit Gaussian process regression with GLGP
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), the training labels.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param a2s A numeric vector, the searching range for bandwidth.
#' @param threshold A double, the threshold ratio for sparse GLGP, defaulting value is 0.01.
#' @param sparse bool, sparse GLGP or not, defaulting value is `TRUE`.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param noise A character vector, taking value in c("same", "different"),
#' indicates whether the noise variances in different locations are different,
#' defaulting value is `same`.
#' @param output_cov Bool, whether to output covariance, defaulting value is `FALSE`.
#'
#' @return `res` A list with three components including `pars`, `posterior` and `Y_pred`,
#' where `Y_pred` is also a list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
#' }
#' @export
fit_gl_regression_gp_rcpp <- function(X, Y, X_new, K, sigma=1e-5, a2s=NULL,
                                 threshold=0.01, sparse=TRUE,
                                 approach ="posterior", noise="same",
                                 output_cov=FALSE) {

  if(is.null(a2s)) {
    a2s = exp(seq(log(0.1),log(10),length.out=10))
  }

  res = fit_gl_regression_gp_cpp(X,Y,X_new,K,sigma,a2s,threshold,sparse,approach,noise,output_cov)

  return(res)

}



#################################################################
# Multinomial Classification
#################################################################


#' Fit Gaussian process logistic multinomial regression with local anchor embedding kernels
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), indicating the labels of multi-classes,
#' `Y` should be continuous integers, such as 0,1,2,...,9.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param r An integer, the number of the nearest neighbor points.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param models A list with four components
#' \describe{
#' \item{subsample}{the method of subsampling in c(`kmeans`, `random`, `minibatchkmeans`), the defaulting value is `kmeans`.}
#' \item{kernel}{the type of kernel to compute cross similarity matrix W, the
#' defaulting value is `lae`.}
#' \item{gl}{the kind of graph Laplacian L, the defaulting value is `cluster-normalized`.}
#' \item{root}{whether to square root eigenvalues of the two steps similarity matrix W,
#' the defaulting value is `TRUE`.}
#' }
#' @param nstart Int, the number of random sets chosen in kmeans.
#'
#' @return A list with two components including `posterior` and `Y_pred`,
#' where `Y_pred` A list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
#' }
#'
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
#' Y_new <- c(rep(0,10),rep(1,10),rep(2,10))
#' s <- 6; r <- 3
#' K <- 5
#' res <- fit_lae_logit_mult_gp_rcpp(X, Y, X_new, s, r, K)
fit_lae_logit_mult_gp_rcpp <- function(X, Y, X_new, s, r, K=-1, sigma=1e-3,
                                       approach="posterior",
                                       models=list(subsample="kmeans",
                                                   kernel="lae",
                                                   gl="cluster-normalized",
                                                   root=TRUE),
                                       nstart=1) {

  # RcppParallel::setThreadOptions(numThreads = RcppParallel::defaultNumThreads()/2)

  Y_pred = fit_lae_logit_mult_gp_cpp(X,Y,X_new,s,r,K,sigma,approach,models,nstart)

  return(Y_pred)
}


#' Fit Gaussian process logistic multinomial regression with square exponential kernels
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), indicating the labels of multi-classes,
#' `Y` should be continuous integers, such as 0,1,2,...,9.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param r An integer, the number of the nearest neighbor points.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param a2s A numeric vector, the searching range for bandwidth.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param models A list with four components
#' \describe{
#' \item{subsample}{the method of subsampling in c(`kmeans`, `random`, `minibatchkmeans`), the defaulting value is `kmeans`.}
#' \item{kernel}{the type of kernel to compute cross similarity matrix W, the
#' defaulting value is `lae`.}
#' \item{gl}{the kind of graph Laplacian L, the defaulting value is `cluster-normalized`.}
#' \item{root}{whether to square root eigenvalues of the two steps similarity matrix W,
#' the defaulting value is `TRUE`.}
#' }
#' @param nstart Int, the number of random sets chosen in kmeans.
#'
#' @return A list with two components including `posterior` and `Y_pred`,
#' where `Y_pred` A list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
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
#' Y_new <- c(rep(0,10),rep(1,10),rep(2,10))
#' s <- 6; r <- 3
#' K <- 5
#' res <- fit_se_logit_mult_gp_rcpp(X, Y, X_new, s, r, K)
fit_se_logit_mult_gp_rcpp <- function(X, Y, X_new, s, r, K=-1, sigma=1e-3, a2s=NULL,
                                       approach="posterior",
                                      models=list(subsample="kmeans",
                                                  kernel="lae",
                                                  gl="cluster-normalized",
                                                  root=TRUE),
                                      nstart=1) {

  if(is.null(a2s)) {
    a2s = exp(seq(log(0.1),log(10),length.out=10))
  }

  Y_pred = fit_se_logit_mult_gp_cpp(X,Y,X_new,s,r,K,sigma,a2s,approach,models,nstart)

  return(Y_pred)
}


#' Fit Gaussian process logistic multinomial regression with Nystrom extension
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), indicating the labels of multi-classes,
#' `Y` should be continuous integers, such as 0,1,2,...,9.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param a2s A numeric vector, the searching range for bandwidth.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param subsample The method of subsampling in c(`kmeans`, `random`, `minibatchkmeans`), the defaulting value is `kmeans`.
#' @param nstart Int, the number of random sets chosen in kmeans.
#'
#' @return A list with two components including `posterior` and `Y_pred`,
#' where `Y_pred` A list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
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
#' Y_new <- c(rep(0,10),rep(1,10),rep(2,10))
#' s <- 6
#' K <- 5
#' res <- fit_nystrom_logit_mult_gp_rcpp(X, Y, X_new, s, K)
fit_nystrom_logit_mult_gp_rcpp <- function(X, Y, X_new, s, K=-1, sigma=1e-3, a2s=NULL,
                                      approach="posterior",
                                      subsample="kmeans",
                                      nstart=1) {
  if(is.null(a2s)) {
    a2s = exp(seq(log(0.1),log(10),length.out=10))
  }

  Y_pred = fit_nystrom_logit_mult_gp_cpp(X,Y,X_new,s,K,sigma,a2s,approach,subsample,nstart)

  return(Y_pred)
}


#' Fit Gaussian process logistic multinomial regression with GLGP
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), indicating the labels of multi-classes,
#' `Y` should be continuous integers, such as 0,1,2,...,9.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param a2s A numeric vector, the searching range for bandwidth.
#' @param threshold A double, the threshold ratio for sparse GLGP, defaulting value is 0.01.
#' @param sparse bool, sparse GLGP or not, defaulting value is `TRUE`.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#'
#' @return A list with two components including `posterior` and `Y_pred`,
#' where `Y_pred` A list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
#' }
#' @export
fit_gl_logit_mult_gp_rcpp <- function(X, Y, X_new, K, sigma=1e-3, a2s=NULL,
                                     threshold=0.01, sparse=TRUE,
                                     approach ="posterior") {
  if(is.null(a2s)) {
    a2s = exp(seq(log(0.1),log(10),length.out=10))
  }

  Y_pred = fit_gl_logit_mult_gp_cpp(X,Y,X_new,K,sigma,a2s,threshold,sparse,approach)

  return(Y_pred)
}


#################################################################
# Binary Classification
#################################################################

#' Fit Gaussian process logistic regression with local anchor embedding kernels
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), count of the positive class.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param r An integer, the number of the nearest neighbor points.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param N A numeric vector with length(m), total count.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param models A list with four components
#' \describe{
#' \item{subsample}{the method of subsampling in c(`kmeans`, `random`, `minibatchkmeans`), the defaulting value is `kmeans`.}
#' \item{kernel}{the type of kernel to compute cross similarity matrix W, the
#' defaulting value is `lae`.}
#' \item{gl}{the kind of graph Laplacian L, the defaulting value is `cluster-normalized`.}
#' \item{root}{whether to square root eigenvalues of the two steps similarity matrix W,
#' the defaulting value is `TRUE`.}
#' }
#' @param output_cov Bool, whether to output covariance, defaulting value is `FALSE`.
#' @param nstart Int, the number of random sets chosen in kmeans.
#'
#' @return `res` A list with three components including `pars`, `posterior` and `Y_pred`,
#' where `Y_pred` is also a list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
#' }
#' @export
#'
#' @examples
#' X0 <- matrix(rnorm(3*3), 3, 3)
#' X1 <- matrix(rnorm(3*3, 5), 3, 3)
#' Y <- c(1,1,1,0,0,0)
#' X <- rbind(X0,X1)
#' X0_new <- matrix(rnorm(10*3),10,3)
#' X1_new <- matrix(rnorm(10*3, 5),10,3)
#' X_new <- rbind(X0_new, X1_new)
#' Y_new <- c(rep(1,10),rep(0,10))
#' s <- 6; r <- 3
#' K <- 5
#' res_pred <- fit_lae_logit_gp_rcpp(X, Y, X_new, s, r, K)
fit_lae_logit_gp_rcpp <- function(X, Y, X_new, s, r, K=-1, N=NULL, sigma=1e-3,
                               approach="posterior",
                               models=list(subsample="kmeans",
                                           kernel="lae",
                                           gl="cluster-normalized",
                                           root=TRUE),
                               output_cov=FALSE,
                               nstart=1) {
  # RcppParallel::setThreadOptions(numThreads = RcppParallel::defaultNumThreads()/2)
  if(is.null(N)) {
    N = rep(1,nrow(X))
  }

  res = fit_lae_logit_gp_cpp(X,Y,X_new,s,r,K,N,sigma,approach,models,output_cov,nstart)

  return(res)

}



#' Fit Gaussian process logistic regression with square exponential kernels
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), count of the positive class.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param r An integer, the number of the nearest neighbor points.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param N A numeric vector with length(m), total count.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param a2s A numeric vector, the searching range for bandwidth.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param models A list with four components
#' \describe{
#' \item{subsample}{the method of subsampling in c(`kmeans`, `random`, `minibatchkmeans`), the defaulting value is `kmeans`.}
#' \item{kernel}{the type of kernel to compute cross similarity matrix W, the
#' defaulting value is `lae`.}
#' \item{gl}{the kind of graph Laplacian L, the defaulting value is `cluster-normalized`.}
#' \item{root}{whether to square root eigenvalues of the two steps similarity matrix W,
#' the defaulting value is `TRUE`.}
#' }
#' @param output_cov Bool, whether to output covariance, defaulting value is `FALSE`.
#' @param nstart Int, the number of random sets chosen in kmeans.
#'
#' @return `res` A list with two components including `pars`, `posterior` and `Y_pred`,
#' where `Y_pred` is also a list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
#' }
#' @export
#'
#' @examples
#' X0 <- matrix(rnorm(3*3), 3, 3)
#' X1 <- matrix(rnorm(3*3, 5), 3, 3)
#' Y <- c(1,1,1,0,0,0)
#' X <- rbind(X0,X1)
#' X0_new <- matrix(rnorm(10*3),10,3)
#' X1_new <- matrix(rnorm(10*3, 5),10,3)
#' X_new <- rbind(X0_new, X1_new)
#' Y_new <- c(rep(1,10),rep(0,10))
#' s <- 6; r <- 3
#' K <- 5
#' res_pred <- fit_se_logit_gp_rcpp(X, Y, X_new, s, r, K)
fit_se_logit_gp_rcpp <- function(X, Y, X_new, s, r, K=-1, N=NULL, sigma=1e-3, a2s=NULL,
                                 approach="posterior",
                                 models=list(subsample="kmeans",
                                             kernel="lae",
                                             gl="cluster-normalized",
                                             root=TRUE),
                                 output_cov=FALSE,
                                 nstart=1) {
  # RcppParallel::setThreadOptions(numThreads = RcppParallel::defaultNumThreads()/2)
  if(is.null(N)) {
    N = rep(1,nrow(X))
  }

  if(is.null(a2s)) {
    a2s = exp(seq(log(0.1),log(10),length.out=10))
  }

  res = fit_se_logit_gp_cpp(X,Y,X_new,s,r,K,N,sigma,a2s,approach,models,output_cov,nstart)

  return(res)
}


#' Fit Gaussian process logistic regression with Nystrom extension
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), count of the positive class.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param N A numeric vector with length(m), total count.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param a2s A numeric vector, the searching range for bandwidth.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param subsample The method of subsampling in c(`kmeans`, `random`, `minibatchkmeans`), the defaulting value is `kmeans`.
#' @param output_cov Bool, whether to output covariance, defaulting value is `FALSE`.
#' @param nstart Int, the number of random sets chosen in kmeans.
#'
#' @return `res` A list with two components including `pars`, `posterior` and `Y_pred`,
#' where `Y_pred` is also a list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
#' }
#' @export
#'
#' @examples
#' X0 <- matrix(rnorm(3*3), 3, 3)
#' X1 <- matrix(rnorm(3*3, 5), 3, 3)
#' Y <- c(1,1,1,0,0,0)
#' X <- rbind(X0,X1)
#' X0_new <- matrix(rnorm(10*3),10,3)
#' X1_new <- matrix(rnorm(10*3, 5),10,3)
#' X_new <- rbind(X0_new, X1_new)
#' Y_new <- c(rep(1,10),rep(0,10))
#' s <- 6
#' K <- 5
#' Y_pred <- fit_nystrom_logit_gp_rcpp(X, Y, X_new, s, K)
fit_nystrom_logit_gp_rcpp <- function(X, Y, X_new, s, K=-1, N=NULL, sigma=1e-3, a2s=NULL,
                                      approach="posterior",
                                      subsample="kmeans",
                                      output_cov=FALSE,
                                      nstart=1) {

  if(K<0) {
    K = s
  }

  if(is.null(N)) {
    N = rep(1,nrow(X))
  }

  if(is.null(a2s)) {
    a2s = exp(seq(log(0.1),log(10),length.out=10))
  }

  res = fit_nystrom_logit_gp_cpp(X,Y,X_new,s,K,N,sigma,a2s,approach,subsample,output_cov,nstart)

  return(res)
}



#' Fit logistic regression with GLGP
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param Y A numeric vector with length(m), count of the positive class.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `NULL`, that is, `K=min(n,s)`.
#' @param N A numeric vector with length(m), total count.
#' @param sigma A non-negative number, the weight coefficient of ridge penalty on H,
#' the defaulting value is 1e-3.
#' @param a2s A numeric vector, the searching range for bandwidth.
#' @param threshold A double, the threshold ratio for sparse GLGP, defaulting value is 0.01.
#' @param sparse bool, sparse GLGP or not, defaulting value is `TRUE`.
#' @param approach A character vector, taking value in c("posterior", "marginal"),
#' decides which objective function to be optimized, defaulting value is `posterior`.
#' @param output_cov Bool, whether to output covariance, defaulting value is `FALSE`.
#'
#' @return `res` A list with two components including `pars`, `posterior` and `Y_pred`,
#' where `Y_pred` is also a list with two components, that is,
#' \describe{
#' \item{train}{A numeric vector with length(m), each element indicates
#' the label in the train data point.}
#' \item{test}{A numeric vector with length(m_new), each element indicates
#' the label in the test data point.}
#' }
#' @export
fit_gl_logit_gp_rcpp <- function(X, Y, X_new, K, N=NULL, sigma=1e-3, a2s=NULL,
                            threshold=0.01, sparse=TRUE,
                            approach ="posterior",
                            output_cov=FALSE) {
  if(is.null(N)) {
    N = rep(1,nrow(X))
  }

  if(is.null(a2s)) {
    a2s = exp(seq(log(0.1),log(10),length.out=10))
  }

  res = fit_gl_logit_gp_cpp(X,Y,X_new,K,N,sigma,a2s,threshold,sparse,approach,output_cov)

  return(res)
}



#' Construct a heat kernel covariance matrix by FLGP
#'
#' @param X Training sample, a (m, d) matrix, each row indicates one point in R^d.
#' @param X_new Testing sample, a (n-m, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param r An integer, the number of the nearest neighbor points.
#' @param t A non-negative number, the heat diffusion time.
#' @param K An integer, the number of used eigenpairs to construct heat kernel,
#' the defaulting value is `-1`, that is, `K=s`.
#' @param models A list, including
#' * `subsample` the method of subsampling, the defaulting value is `kmeans`.
#' * `kernel` the type of kernel to compute cross similarity matrix W, the
#' defaulting value is `lae`.
#' * `gl` the kind of graph Laplacian L, the defaulting value is `rw`.
#' * `root` whether to square root eigenvalues of the two steps similarity matrix W,
#' the defaulting value is `FALSE`.
#' @param nstart An integer, the number of random sets chosen in kmeans,
#' the defaulting value is `1`.
#'
#' @return A numeric matrix with dim (n,m), the heat kernel covariance matrix estimated
#' by FLGP.
#'
#' @export
#'
#' @examples
#' X <- matrix(rnorm(2*2), 2, 2)
#' X_new <- matrix(rnorm(3*2),3,2)
#' s <- 2; r <- 2; t <- 1
#' heat_kernel_covariance_rcpp(X, X_new, s, r, t)
heat_kernel_covariance_rcpp <- function(X, X_new, s, r, t, K=-1,
                            models=list(subsample="kmeans",
                                        kernel="lae",
                                        gl="cluster-normalized",
                                        root=TRUE),
                            epsilon=0.1,
                            nstart=1) {
  H = heat_kernel_covariance_cpp(X,X_new,s,r,t,K,models,nstart,epsilon)

  return(H)
}
