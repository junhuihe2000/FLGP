#' Subsample in the domain
#'
#' @param X Original sample, a (n, d) matrix, each row indicates one point in R^d.
#' @param s An integer indicating the number of the subsampling.
#' @param method How to subsample, characters in c("kmeans", "random"),
#' including kmeans and random selection, the
#' defaulting subsampling method is kmeans.
#'
#' @return A subsampling, a (s, d) or (s, d+1) matrix, each row indicates one point in R^d,
#' where the d+1 column indicates the number of points in each cluster if it exists.
#' @export
#'
#' @examples
#' X <- matrix(rnorm(100*3), nrow=100, ncol=3)
#' s <- 10
#' U = subsample(X, s, method = "kmeans")
subsample <- function(X, s, method = "kmeans") {
  stopifnot(is.matrix(X), abs(s-round(s))<.Machine$double.eps^0.5)
  if(method == "kmeans") {
    cluster_kmeans = stats::kmeans(X, s, iter.max = 20, nstart = 10)
    U = cbind(cluster_kmeans$centers, size=cluster_kmeans$size)
  } else if(method == "random") {
    U = X[sample.int(nrow(X), s), ]
    if(s == 1) {
      U = matrix(U, nrow=1)
    }
  } else {
    stop("Error: the subsample method is not supported!")
  }
  return(U)
}
