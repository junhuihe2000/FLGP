test_that("truncated_svd() computes the non-trivial spetrum of A%*%t(A)", {
  A = matrix(rnorm(100*10),100,10)
  spec_trun = truncated_SVD(A)
  vectors = spec_trun$vectors
  expect_true(Matrix::norm(Matrix::colScale(vectors, spec_trun$values)%*%Matrix::t(vectors)-A%*%Matrix::t(A))
              <1e-3)
})
