test_that("spectrum_from_Z() computes the spectrum of W from Z", {
  Z = matrix(rnorm(5,3), 5, 3)
  W = Z%*%diag(colSums(Z)^{-1})%*%t(Z)
  pairs = spectrum_from_Z(Z)
  expect_true(Matrix::norm(W%*%pairs$vectors-pairs$vectors%*%diag(pairs$values))<1e-3)
})
