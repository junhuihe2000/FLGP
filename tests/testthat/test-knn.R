test_that("which_minn() finds indexes of the first r smallest elements", {
  z = rnorm(100)
  r = 10
  expect_equal(which_minn(z, r), order(z)[1:r])
})


test_that("which_minn() errors when z is not a numeric vector or r is not an integer", {
  expect_error(which_minn(matrix(0,1,1), 1))
  expect_error(which_minn(c(1,2,3), 0.5))
})


test_that("KNN() errors when X or U is not a matrix, or r is not an integer,
          or columns between X and U are not equal", {
            expect_error(KNN(1,matrix(1),1))
            expect_error(KNN(matrix(1),1,1))
            expect_error(KNN(matrix(1),matrix(1),1.1))
            expect_error(KNN(matrix(1),matrix(c(1,2),1,2),1))
          })


test_that("KNN() finds the indexes of the k-nearest neighbor points", {
  expect_equal(KNN(X=matrix(c(0,0,0,3),nrow=2),
                   U=matrix(c(1,0,0,2),nrow=2),
                   r=1,
                   distance="Euclidean"),
               list(1,2))
})


