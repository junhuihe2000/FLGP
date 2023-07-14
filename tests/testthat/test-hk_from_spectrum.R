test_that("HK_from_spectrum() constructs heat kernel covariance matrix from spectrums", {
  Z = matrix(rnorm(3*3),3,3)
  X = Z%*%t(Z)
  eigens = eigen(X)
  m = 2; K = 2; t = 1
  cov_hk = eigens$vectors[,1:K]%*%diag(exp(-t*(1-eigens$values[1:K])))%*%t(eigens$vectors[1:m,1:K])
  expect_true(Matrix::norm(cov_hk - HK_from_spectrum(eigens, K, t, NULL, c(1:m)))<1e-3)
})
