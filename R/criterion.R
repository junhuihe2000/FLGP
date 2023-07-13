#' Compute classification error rates
#'
#' @param Y_new Labels of new samples.
#' @param Y_pred Predicted labels of new samples.
#'
#' @return A double, the error rate.
#' @export
#'
#' @examples
#' Y_new <- c(1,1,0,0)
#' Y_pred <- c(1,0,1,0)
#' error_rate(Y_new, Y_pred)
error_rate <- function(Y_new, Y_pred) {
  if(is.vector(Y_new)) {
    stopifnot(is.vector(Y_pred), length(Y_new)==length(Y_pred))
    n = length(Y_new)
    return(sum(Y_new!=Y_pred)/n)
  }
  else if(is.matrix(Y_new)) {
    stopifnot(is.matrix(Y_pred), all(dim(Y_new)==dim(Y_pred)))
    n = nrow(Y_new)
    res = vapply(c(1:n), function(i){
      return(which(Y_new[i,]==1)!=which(Y_pred[i,]==1))
    }, FUN.VALUE = TRUE)
    return(sum(res)/n)
  }
  else {
    stop("The format of Y is not supported!")
  }
}
