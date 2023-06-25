# r nearest neighbor representative points
# para:
#       X: (n, d) matrix, each row indicates one original point
#       U: (s, d) matrix, each row indicates one representative point
#       r: integer, r nearest neighbor
# return value:
#       ind_knn: list with length n, each component indicates
#               indexes of r nearest neighbor for x_i, allowing replicate
KNN <- function(X, U, r) {
  distance = t(apply(X=X, MARGIN=1, FUN=euclidean_distance, U=U))
  ind_knn = apply(X=distance, MARGIN=1, FUN=which_minn, r=r, simplify=FALSE)
  return(ind_knn)
}

# compute square Euclidean distance between x and U
# para:
#       x: d vector, one original point
#       U: (s, d) matrix, each row indicates one representative point
# return value:
#       ed2: s vector, square Euclidean distance
euclidean_distance <- function(x, U) {
  ed2 = rowSums(t(x-t(U))^2)
  return(ed2)
}

# find indexes of the first k smallest elements in arrays, allowing replication
# para:
#     z: vector of s doubles
#     r: integer, the first r smallest elements
# return value:
#     vector of r indexes
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
