test_that("which_minn() finds indexes of the first r smallest elements", {
  expect_equal(which_minn(c(1,3,2), 2), c(1,3))
})

test_that("euclidean_distance() computes Euclidean distances between x and U", {
  expect_equal(euclidean_distance(x=c(0,0),
                                  U=matrix(c(1,0,0,2), nrow=2)),
               c(1,2))
})

test_that("KNN() finds the indexes of the k-nearest neighbor points", {
  expect_equal(KNN(X=matrix(c(0,0,0,3),nrow=2),
                   U=matrix(c(1,0,0,2),nrow=2),
                   r=1,
                   distance="Euclidean"),
               list(1,2))
})
