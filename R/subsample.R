# sub-sampling

# para:
#     X: (n, d) matrix, original sample, each row indicates one point in R^d
#     s: integer, the number of the sub-sample
#     methods = c("kmeans", "random"), how to sub-sample
# return value:
#     U: (s, d) matrix, sub-sample, each row indicates one "new" point in R^d
subsample <- function(X, s, method = "kmeans") {
  if(method == "kmeans") {
    U = kmeans(X, s, iter.max = 20, nstart = 10)$centers
  } else if(method == "random") {
    U = X[sample.int(nrow(X), s), ]
  } else {
    stop("The subsample method is not supported!")
  }
  return(U)
}
