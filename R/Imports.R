# R interfaces for importing eigs_sym from RSpectra
eigs_sym_r <- function(A, k) {
  return(RSpectra::eigs_sym(A, k))
}
