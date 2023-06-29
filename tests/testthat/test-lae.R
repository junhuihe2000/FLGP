test_that("v_to_z() projects a vector to the simplex", {
  expect_equal(v_to_z(c(0.2,0.3,0.5)), c(0.2,0.3,0.5))
})

test_that("local_anchor_embedding() returns convex coefficients", {
  x = rnorm(3)
  U = matrix(rnorm(3*3),3,3)
  z = local_anchor_embedding(list("x"=x,"U"=U))
  expect_true(abs(sum(z)-1)<.Machine$double.eps^0.5)
  for(i in 1:3) {
    expect_true(z[i]>=0)
  }
})
