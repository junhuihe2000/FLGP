#' Local anchor embedding
#'
#' @param X A numeric matrix with dim (n,d), original sample,
#' each row indicates one original point in R^d.
#' @param U A numeric matrix with dim (s,d) or (s,d+1), sub-sample,
#' each row indicates one representative point in R^d,
#' where the d+1 column indicates the number of points in each cluster if it exists.
#' @param r An integer, the number of the nearest neighbor points.
#' @param gl A character vector in c("rw", "normalized", "cluster-normalized"),
#' indicates how to construct the stochastic transition matrix. "rw" means random walk,
#' "normalized" means normalized random walk, "cluster-normalized" means
#' normalized random walk with cluster membership re-balance. The defaulting gl
#' is "rw".
#'
#' @return A numeric 'sparse' matrix with dim (n,s) and n*r non-zero entries, r << s,
#' the stochastic transition matrix from X to U.
#' @export
#'
#' @examples
#' X <- matrix(rnorm(10*3),10,3)
#' r <- 3
#' U_1 <- matrix(rnorm(5*3),5,3)
#' LAE(X, U_1, r)
#' U_2 <- matrix(rnorm(5*4),5,4)
#' LAE(X, U_2, r)
LAE <- function(X, U, r=3L, gl="rw") {
  stopifnot(is.matrix(X), is.matrix(U), abs(r-round(r))<.Machine$double.eps^0.5)
  stopifnot(ncol(U)-ncol(X)==0||ncol(U)-ncol(X)==1)
  n = nrow(X); s = nrow(U); d = ncol(X)
  ind_knn = KNN(X, U[,1:d], r)
  lae_units = lapply(c(1:n), function(i){return(list("x"=X[i, ], "U"=U[ind_knn[[i]], 1:d]))})

  chk <- Sys.getenv("_R_CHECK_LIMIT_CORES_", "")
  if (nzchar(chk) && chk == "TRUE") {
    # use 2 cores in CRAN/Travis/AppVeyor
    num_workers <- 2L
  } else {
    # use all cores in devtools::test()
    num_workers <- parallel::detectCores() - 1
  }
  cl = parallel::makeCluster(num_workers)
  Z_list = parallel::parLapply(cl, lae_units,
                                function(unit) {return(local_anchor_embedding(unit$x, unit$U))})
  parallel::stopCluster(cl)

  Z = Matrix::sparseMatrix(i=rep(c(1:n), each=r),
                           j=unlist(ind_knn),
                           x=unlist(Z_list))

  if(gl=="rw") {}
  else if(gl=="normalized") {
    Z_norm = Matrix::colScale(Z, Matrix::colSums(Z)^{-1})
    Z = Matrix::rowScale(Z_norm, Matrix::rowSums(Z_norm)^{-1})
  } else if(gl=="cluster-normalized") {
    stopifnot(ncol(U)==(ncol(X)+1))
    Z_norm = Matrix::colScale(Z, Matrix::colSums(Z)^{-1})
    Z_cl = Matrix::colScale(Z_norm, U[,d+1])
    Z = Matrix::rowScale(Z_cl, Matrix::rowSums(Z_cl)^{-1})
  } else {
    stop("Error: the type of graph Laplacian is not supported!")
  }
  return(Z)
}



#' Local anchor embedding of one single point by
#' gradient descent projection with Nesterov's methods
#'
#' @param x A numeric vector with length d, indicates the single point to be embedded.
#' @param U A numeric matrix with dimension (r, d), the columns of U
#' are equal to the length of x, including KNN reference points.
#'
#' @return A numeric vector with the length r, convex combination coefficients.
#' @export
#'
#' @examples
#' x <- rnorm(3)
#' U <- matrix(rnorm(3*3),3,3)
#' local_anchor_embedding(x, U)
local_anchor_embedding <- function(x, U) {
  stopifnot(is.vector(x), is.matrix(U), length(x)==ncol(U))
  x = matrix(x, ncol=1); U = t(U)
  r = ncol(U); d = nrow(U)

  # initialize
  z_prev = matrix(1/r,r,1); z_curr = z_prev
  delta_prev = 0; delta_curr = 1
  beta_curr = 1
  # stop criterion
  tol = 1e-5; t = 0; T = 100

  # useful inter-quantity
  UtU = crossprod(U)

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
    repeat {
      # backtrack
      beta = 2^j*beta_curr
      # gradient descent
      v_tilde = v - 1/beta*grad_v
      # projection
      z = matrix(v_to_z(v_tilde), ncol = 1)
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
  return(as.vector(z_curr))
}




#' Simplex projection into convex combination coefficients
#'
#' @param v A numeric vector to be projected.
#'
#' @return A numeric vector of convex combination coefficients with
#' the same length of v.
#' @export
#'
#' @examples
#' v <- rnorm(3)
#' v_to_z(v)
v_to_z <- function(v) {
  v_desc = sort(v, decreasing=TRUE)
  r = length(v)
  v_star = v_desc - (cumsum(v_desc)-1)/c(1:r)
  for(rho in r:1) {
    if(v_star[rho]>0) {break}
  }
  theta = (sum(v_desc[1:rho])-1)/rho
  z = vapply(v-theta, function(x){return(max(x,0))}, FUN.VALUE = 0.5)
  return(z)
}
