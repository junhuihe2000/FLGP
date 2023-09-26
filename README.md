
<!-- README.md is generated from README.Rmd. Please edit that file -->

# FLAG

<!-- badges: start -->
<!-- badges: end -->

The goal of FLAG is to provides algorithms to make inference in the
large scale data by the approximate heat kernel Gaussian process. The
fast computation speed benefits from subsampling, constructing graph
Laplacian through the transition decomposition and the truncated SVD.
Then the prior geometric covariance matrix is calculated by the spectral
properties of heat kernels. The package also includes Gaussian process
regression(GPR), Gaussian process logistic regression and multinomial
logistic regression with Polya-Gamma samplers(GPC).

## Installation

You can install the development version of FLAG from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("junhuihe2000/FLAG")
```

## Example

These are some basic examples which show you how to solve a common
problem:

### Gaussian process classification(GPC)

``` r
library(ggplot2)
library(FLAG)
## basic example code

set.seed(1234)
```

``` r
## generate samples
n = 4800; n_each = 800; d = 3
thetas = runif(n, 0, 2*pi)
X = matrix(0, nrow=n, ncol=2); Y = matrix(0, nrow=n, ncol=1)
X[,1] = cos(thetas); X[,2] = sin(thetas)
for(i in 0:5) {
  X[(i*n_each+1):((i+1)*n_each),] = (0.5+0.1*i)*X[(i*n_each+1):((i+1)*n_each),]
  Y[(i*n_each+1):((i+1)*n_each),] = as.numeric((-1)^i>0)
}

## process data
X = scale(X, center = TRUE, scale = TRUE)
X = X / sqrt(ncol(X))

## divide the training set and the testing set
m = 100
train.index = sample.int(n, m); test.index = c(1:n)[-train.index]
train.data = X[train.index,]; test.data = X[test.index,]
train.label = Y[train.index,]; test.label = Y[test.index,]


## plot
ggplot() + geom_point(aes(X[,1],X[,2],color=factor(Y)), size=0.6, alpha=0.6) +
  scale_color_manual(values = c("red", "orange")) + 
  theme(legend.position="none", 
        axis.title.x=element_blank(), axis.title.y=element_blank()) +
  ggtitle("Classification")
```

<img src="man/figures/README-unnamed-chunk-2-1.png" width="100%" />

``` r
## FLAG

## set hyper-parameters
s = 600; r = 3; K = 100
models = list(subsample="kmeans", kernel="lae", gl="cluster-normalized", root=TRUE)
```

``` r
## FLAG with the square exponential kernel and k-means subsampling
cat("SKFLAG:\n")
#> SKFLAG:
t1 = Sys.time()
y_skflag.torus = fit_se_logit_gp_rcpp(train.data, train.label, test.data, s, r, K, models = models)
t2 = Sys.time()
print(t2-t1)
#> Time difference of 19.93066 secs
err_skflag.torus = sum((test.label!=y_skflag.torus)^2)/(n-m)
cat("The error rate of SKFLAG is",err_skflag.torus,".\n")
#> The error rate of SKFLAG is 0 .
```

``` r
## FLAG with local anchor embedding and k-means subsampling
cat("LKFLAG:\n")
#> LKFLAG:
t3 = Sys.time()
y_lkflag.torus = fit_lae_logit_gp_rcpp(train.data, train.label, test.data, s, r, K, models = models)
t4 = Sys.time()
print(t4-t3)
#> Time difference of 3.678423 secs
err_lkflag.torus = sum((test.label-y_lkflag.torus)^2)/(n-m)
cat("The error rate of LKFLAG is",err_lkflag.torus,".\n")
#> The error rate of LKFLAG is 0.02702128 .
```

### Gaussian process regression(GPR)

``` r
n = 4000
theta = runif(n,0,8*pi)
X = cbind((theta+4)^(0.7)*cos(theta), (theta+4)^(0.7)*sin(theta))
Y = 3*sin(theta/10) + 3*cos(theta/2) + 4*sin(4*theta/5)

ggplot() + geom_point(aes(X[,1],X[,2],color=Y)) +
  scale_color_gradientn(colours = rainbow(10)) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  ggtitle("Regression")
```

<img src="man/figures/README-unnamed-chunk-6-1.png" width="100%" />

``` r
m = 100
train.index = sample.int(n,m); test.index = c(1:n)[-train.index]
X.train = X[train.index,]; Y.train = Y[train.index]
X.test = X[test.index,]; Y.test = Y[test.index]
```

``` r
s = 500; r = 3; K = 100
```

``` r
Y_pred_skflag = fit_se_regression_gp_rcpp(X.train,Y.train,X.test,s,r,K,models=models)
```

``` r
rmse_skflag.spiral = sqrt(sum((Y.test-Y_pred_skflag)^2)/(n-m))
cat("The RMSE of SKFLAG is",rmse_skflag.spiral,".\n")
#> The RMSE of SKFLAG is 0.3497363 .
```

``` r
ggplot() + geom_point(aes(X.test[,1],X.test[,2],color=Y_pred_skflag)) +
  scale_color_gradientn(colours = rainbow(10)) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  ggtitle("SKFLAG in Regression")
```

<img src="man/figures/README-unnamed-chunk-11-1.png" width="100%" />

``` r
Y_pred_lkflag = fit_lae_regression_gp_rcpp(X.train,Y.train,X.test,s,r,K,models=models)
```

``` r
rmse_lkflag.spiral = sqrt(sum((Y.test-Y_pred_lkflag)^2)/(n-m))
cat("The RMSE of LKFLAG is",rmse_lkflag.spiral,".\n")
#> The RMSE of LKFLAG is 0.5704005 .
```

``` r
ggplot() + geom_point(aes(X.test[,1],X.test[,2],color=Y_pred_lkflag)) +
  scale_color_gradientn(colours = rainbow(10)) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  ggtitle("LKFLAG in Regression")
```

<img src="man/figures/README-unnamed-chunk-14-1.png" width="100%" />
