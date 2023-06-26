#' Subsample in the domain
#'
#' @param X Original sample, a (n, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param method How to subsample, including kmeans and random selection, the
#' defaulting subsampling method is kmeans.
#'
#' @return U Subsampling, a (s, d) matrix, each row indicates one point in R^d.
#' @export
#'
#' @examples
#' X <- matrix(rnorm(100*3), nrow=100, ncol=3)
#' s <- 10
#' U = subsample(X, s, method = "kmeans")
subsample <- function(X, s, method = "kmeans") {
  stopifnot(is.matrix(X), abs(s-round(s))<.Machine$double.eps^0.5)
  if(method == "kmeans") {
    U = stats::kmeans(X, s, iter.max = 20, nstart = 10)$centers
  } else if(method == "random") {
    U = X[sample.int(nrow(X), s), ]
    if(s == 1) {
      U = matrix(U, nrow=1)
    }
  } else {
    stop("The subsample method is not supported!")
  }
  return(U)
}
