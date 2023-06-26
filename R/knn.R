#' k-nearest neighbor reference points
#'
#' @param X Original points, a (n,d) matrix, each row indicates one original point.
#' @param U Reference points, a (s,d) matrix, each row indicates one reference point.
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
#' U <- subsample(X, 10)
#' r <- 3
#' distance <- "Euclidean"
#' KNN(X, U, r, distance)
KNN <- function(X, U, r, distance="Euclidean") {
  stopifnot(is.matrix(X), is.matrix(U), abs(r-round(r))<.Machine$double.eps^0.5, ncol(X)==ncol(U))
  if(distance=="Euclidean") {
    distances = t(apply(X=X, MARGIN=1, FUN=euclidean_distance, U=U))
  } else {
    stop("The distance of KNN is not supported!")
  }
  ind_knn = apply(X=distances, MARGIN=1, FUN=which_minn, r=r, simplify=FALSE)
  return(ind_knn)
}



#' Euclidean distance between one point and reference points
#'
#' @param x One original point, a vector of length d.
#' @param U Reference points, a (s,d) matrix, each row indicates one reference point.
#'
#' @return The Euclidean distance between x and U, a vector of length s.
#' @export
#'
#' @examples
#' x <- rnorm(3)
#' U <- matrix(rnorm(30), nrow=10, ncol=3)
#' euclidean_distance(x, U)
euclidean_distance <- function(x, U) {
  stopifnot(is.vector(x), is.matrix(U), length(x)==ncol(U))
  ed2 = sqrt(rowSums(t(x-t(U))^2))
  return(ed2)
}



#' Find indexes of the first r smallest elements in arrays based on Bubblesorting
#'
#' @param z a vector of length s, including elements to be sorted.
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
  n = length(z)
  ind_z = c(1:n)
  for(i in 1:r) {
    for (j in n:(i+1)) {
      if(z[ind_z[j]]<z[ind_z[j-1]]) {
        tem = ind_z[j]; ind_z[j] = ind_z[j-1]; ind_z[j-1] = tem
      }
    }
  }
  return(ind_z[1:r])
}
