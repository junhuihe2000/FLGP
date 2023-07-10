#' Gibbs sampler for Polya-gamma auxiliary variables
#'
#' @param aug_data An augmented data for polya-gamma sampling.
#' @param N_sample An integer, the length of the sampling chain.
#'
#' @return `aug_data` The updated augmented data.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(3*3),3,3)
#' C <- A%*%t(A)
#' Y <- matrix(sample.int(3*3 ,replace=TRUE),3,3)
#' aug_data = AugmentedData(C,Y)
#' PG_Gibbs_sampler(aug_data, 100)
PG_Gibbs_sampler <- function(aug_data=AugmentedData(), N_sample = 100) {
  for(i in c(1:N_sample)) {
    aug_data$f = resample_f(aug_data)
    aug_data$omega = resample_omega(aug_data)
  }
  return(aug_data)
}
