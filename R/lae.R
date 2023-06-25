# Local anchor embedding

# Local anchor embedding of original sample
# para:
#     X: (n, d) matrix, original sample, each row indicates one original point
#     U: (s, d) matrix, sub-sample, each row indicates one representative point
#     r: integer, the number of the nearest neighbors
# return value:
#     Z: (n, s) matrix, stochastic transition matrix from X to U
LAE <- function(X, U, r) {
  n = nrow(X); s = nrow(U); d = ncol(X)
  ind_knn = KNN(X, U, r)
  lae_units = lapply(c(1:n), function(i){return(list("x"=X[i, ], "U"=U[ind_knn[[i]], ]))})
  Z_list = lapply(lae_units, local_anchor_embedding)
  Z = matrix(0, nrow=n, ncol=s) # Z is a sparse matrix
  for(i in 1:n) {
    Z[i, ind_knn[[i]]] = Z_list[[i]]
  }
  return(Z)
}



# Local anchor embedding of one single point
# Gradient descent projection with Nesterov's methods
# para:
#     data: a list of x and U, where x is d-column vector, U is (r, d) matrix
# return value:
#     z_curr: r-column vector, convex combination coefficients
local_anchor_embedding <- function(data) {
  x = matrix(data$x, ncol=1); U = t(data$U)
  r = ncol(U); d = nrow(U)

  # initialize
  z_prev = matrix(rep(1/r,r),ncol=1); z_curr = z_prev
  delta_prev = 0; delta_curr = 1
  beta_curr = 1
  # stop criterion
  tol = 1e-5; t = 0; T = 100

  # useful inter-quantity
  UtU = t(U)%*%U

  # Nesterov momentum method
  while(t<T) {
    t = t + 1
    # momentum exponential ratio
    alpha = (delta_prev-1)/delta_curr
    # sliding average
    v = z_curr + alpha*(z_curr-z_prev)
    # value
    g_v = sum((x-U%*%v)^2)/2
    # gradient
    grad_v = UtU%*%v-t(U)%*%x
    # backtracking search
    j = 0
    while(TRUE) {
      # backtrack
      beta = 2^j*beta_curr
      # gradient descent
      v_tilde = v - 1/beta*grad_v
      # projection
      z = v_to_z(v_tilde)
      # update condition
      g_z = sum((x-U%*%z)^2)/2
      g_tilde = g_v + t(grad_v)%*%(z-v) + beta*sum((z-v)^2)/2
      if(g_z<=g_tilde) {
        beta_curr = beta
        z_prev = z_curr
        z_curr = z
        break
      }
      j = j + 1
    }
    delta_prev = delta_curr
    delta_curr = (1+sqrt(1+4*delta_curr^2))/2
    # repeat until convergence
    if(sum((z_curr-z_prev)^2)<tol) {break}
  }
  # cat("iterations t =",t,"\n")
  return(z_curr)
}



# Simplex projection
# para:
#     v: r column vector
# return value:
#     z: r column vector, each entry non-negative, convex combination coefficients
v_to_z <- function(v) {
  v_desc = sort(v, decreasing=TRUE)
  r = length(v)
  v_star = v_desc - (cumsum(v_desc)-1)/c(1:r)
  for(rho in r:1) {
    if(v_star[rho]>0) {break}
  }
  theta = (sum(v_desc[1:rho])-1)/rho
  z = sapply(v-theta, function(x){return(max(x,0))})
  return(matrix(z, ncol=1))
}
