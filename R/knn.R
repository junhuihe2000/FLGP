#' k-nearest neighbor reference points
#'
#' @param X Original points, a (n,d) matrix, each row indicates one original point.
#' @param U Reference points, a (s,d) or (s,d+1) matrix, each row indicates one reference point.
#' @param r The number of k-nearest neighbor points, an integer.
#' @param distance The distance to compute k-nearest neighbor points, characters in c("Euclidean", "geodesic"),
#'  including Euclidean distance and geodesic distance, the defaulting distance
#'  is Euclidean distance.
#'
#' @return The indexes of KNN, a list with length n, each component of the list is a vector of length r,
#'  indicating the indexes of KNN for the corresponding original point based on the chosen distance.
#' @export
#'
#' @examples
#' X <- matrix(rnorm(300), nrow=100, ncol=3)
#' U <- matrix(rnorm(30), nrow=10, ncol=3)
#' r <- 3
#' distance <- "Euclidean"
#' KNN(X, U, r, distance)
KNN <- function(X, U, r, distance="Euclidean") {
  stopifnot(is.matrix(X), is.matrix(U), abs(r-round(r))<.Machine$double.eps^0.5, ncol(X)==ncol(U))
  n = nrow(X)
  if(distance=="Euclidean") {
    distances = rowSums(X^2)-2*X%*%t(U) + matrix(rowSums(U^2), n, nrow(U), byrow = TRUE)
  } else {
    stop("Error: the distance of KNN is not supported!")
  }
  distances_list = lapply(c(1:n), function(i) {return(distances[i,])})
  chk <- Sys.getenv("_R_CHECK_LIMIT_CORES_", "")

  if (nzchar(chk) && chk == "TRUE") {
    # use 2 cores in CRAN/Travis/AppVeyor
    num_workers <- 2L
  } else {
    # use all cores in devtools::test()
    num_workers <- parallel::detectCores() - 4
  }

  cl = parallel::makeCluster(num_workers)
  ind_knn = parallel::parLapply(cl, distances_list, which_minn, r=r)
  on.exit(parallel::stopCluster(cl))
  return(ind_knn)
}



#' Find indexes of the first r smallest elements in arrays based on Bubblesorting
#'
#' @param z A vector of length s, including elements to be sorted.
#' @param r The number of the smallest elements, an integer.
#'
#' @return A vector of length r, indicating the indexes of the first r smallest elements.
#' @export
#'
#' @examples
#' z <- c(1,3,2)
#' r <- 2
#' which_minn(z, r)
which_minn <- function(z, r) {
  stopifnot(is.vector(z), abs(r-round(r))<.Machine$double.eps^0.5)
  return(order(z)[1:r])
}
