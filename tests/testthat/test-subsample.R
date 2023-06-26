test_that("subsample() subsamples from the original sample", {
  expect_equal(subsample(X=matrix(c(1,2,3),nrow=1,ncol=3), s=1, method = "random") ,
               c(1,2,3))
})
