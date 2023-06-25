# sub-sampling

# para:
#     X: (n, d) matrix, original sample, each row indicates one point in R^d
#     s: integer, the number of the sub-sample
# return value:
#     U: (s, d) matrix, sub-sample, each row indicates one "new" point in R^d
subsample <- function(X, s) {
  U = kmeans(X, s, iter.max = 20, nstart = 10)$centers
  return(U)
}
