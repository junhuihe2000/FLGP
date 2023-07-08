#' Stick breaking transform from f to pi
#'
#' @param f A numeric matrix with dim(M,J-1), latent function.
#'
#' @return `pi` A numeric matrix with dim(M, J), class probability.
#' @export
#'
#' @examples
#' f <- matrix(rnorm(3*3),3,3)
#' f_to_pi(f)
f_to_pi <- function(f) {
  stopifnot(methods::is(f, "matrix")||methods::is(f, "Matrix"))
  M = nrow(f); J = ncol(f) + 1
  pi = matrix(0, M, J)

  sticks = rep(1, M)

  for(j in c(1:(J-1))) {
    pi[,j] = ilogit(f[,j]) * sticks
    sticks = sticks - pi[,j]
  }
  pi[,J] = sticks
  if(sum(abs(Matrix::rowSums(pi)-1))>.Machine$double.eps^0.5) {
    stop("Error: results of f_to_pi() are not valid probabilities!")
  }

  return(pi)
}



#' Stick breaking transform from pi to f
#'
#' @param pi A numeric matrix with dim(M,J), class probability.
#'
#' @return `f` a numeric matrix with dim(M,J-1), latent function.
#' @export
#'
#' @examples
#' pi <- matrix(abs(rnorm(3*3))+1,3,3)
#' pi <- Matrix::rowScale(pi, Matrix::rowSums(pi)^{-1})
#' pi_to_f(pi)
pi_to_f <- function(pi) {
  stopifnot(methods::is(pi, "matrix")||methods::is(pi, "Matrix"))
  M = nrow(pi); J = ncol(pi)

  f = matrix(0, M, J-1)

  sticks = rep(1, M)

  for(j in 1:(J-1)) {
    f[,j] = logit(pi[,j]/sticks)
    sticks = sticks - pi[,j]
  }
  if(sum(abs(sticks-pi[,J]))>.Machine$double.eps^0.5) {
    stop("Error: pi is not a valid probability!")
  }
  return(f)
}


# inverse logit link function
ilogit <- function(x) {
  return(1/(1+exp(-x)))
}

# logit link function
logit <- function(p) {
  return(log(p/(1-p)))
}
