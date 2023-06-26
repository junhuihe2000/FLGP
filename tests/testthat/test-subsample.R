test_that("subsample() subsamples from the original sample", {
  expect_equal(subsample(X=matrix(c(1,2,3),nrow=1,ncol=3), s=1, method = "random") ,
               matrix(c(1,2,3),nrow=1,ncol=3))
})

test_that("subsample() errors if X is not a matrix", {
  expect_error(subsample(c(1,2,3), s=1))
})

test_that("subsample() errors if s is not an integer", {
  expect_error(subsample(matrix(c(1,2,3),nrow=1,ncol=3), s=0.5))
})

test_that("subsample() errors if method is not in c('kmeans', 'random')", {
  expect_error(subsample(matrix(c(1,2,3),nrow=1,ncol=3), s=1, method="randomforest"))
})
