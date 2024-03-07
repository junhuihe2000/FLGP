#ifndef FIT_H
#define FIT_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>



// Fit Gaussian process regression with local anchor embedding kernels
// [[Rcpp::export(fit_lae_regression_gp_cpp)]]
Rcpp::List fit_lae_regression_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                     int s, int r, int K,
                                     double sigma, std::string approach, std::string noise,
                                     Rcpp::List models,
                                     bool output_cov,
                                     int nstart);

// Fit Gaussian process regression with the square exponential kernels
// [[Rcpp::export(fit_se_regression_gp_cpp)]]
Rcpp::List fit_se_regression_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                    int s, int r, int K,
                                    double sigma, std::vector<double> a2s, std::string approach,
                                    std::string noise,
                                    Rcpp::List models,
                                    bool output_cov,
                                    int nstart);

// Fit Gaussian process regression with the nystrom extension
// [[Rcpp::export(fit_nystrom_regression_gp_cpp)]]
Rcpp::List fit_nystrom_regression_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                         int s, int K,
                                         double sigma, std::vector<double> a2s, std::string approach,
                                         std::string noise,
                                         Rcpp::List models,
                                         bool output_cov,
                                         int nstart);

// Fit Gaussian process regression with the graph Laplacian
// [[Rcpp::export(fit_gl_regression_gp_cpp)]]
Rcpp::List fit_gl_regression_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                    int K,
                                    double sigma, std::vector<double> a2s,
                                    double threshold, bool sparse,
                                    std::string approach, std::string noise,
                                    Rcpp::List models,
                                    bool output_cov);

// Fit Gaussian process logistic regression with local anchor embedding kernels
// [[Rcpp::export(fit_lae_logit_gp_cpp)]]
Rcpp::List fit_lae_logit_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                int s, int r, int K, Rcpp::NumericVector N_train,
                                double sigma, std::string approach,
                                Rcpp::List models,
                                bool output_cov,
                                int nstart);


// Fit Gaussian process logistic multinomial regression with local anchor embedding kernels
// [[Rcpp::export(fit_lae_logit_mult_gp_cpp)]]
Rcpp::List fit_lae_logit_mult_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                     int s, int r, int K,
                                     double sigma, std::string approach,
                                     Rcpp::List models,
                                     int nstart);



// Fit Gaussian process logistic regression with square exponential kernels
// [[Rcpp::export(fit_se_logit_gp_cpp)]]
Rcpp::List fit_se_logit_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                               int s, int r, int K, Rcpp::NumericVector N_train,
                               double sigma, std::vector<double> a2s, std::string approach,
                               Rcpp::List models,
                               bool output_cov,
                               int nstart);


// Fit Gaussian process logistic multinomial regression with square exponential kernels
// [[Rcpp::export(fit_se_logit_mult_gp_cpp)]]
Rcpp::List fit_se_logit_mult_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                    int s, int r, int K,
                                    double sigma, std::vector<double> a2s, std::string approach,
                                    Rcpp::List models,
                                    int nstart);

// Fit Gaussian process logistic regression with Nystrom extension
// [[Rcpp::export(fit_nystrom_logit_gp_cpp)]]
Rcpp::List fit_nystrom_logit_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                    int s, int K, Rcpp::NumericVector N_train,
                                    double sigma, std::vector<double> a2s, std::string approach,
                                    Rcpp::List models,
                                    bool output_cov,
                                    int nstart);

// Fit Gaussian process logistic multinomial regression with Nystrom extension
// [[Rcpp::export(fit_nystrom_logit_mult_gp_cpp)]]
Rcpp::List fit_nystrom_logit_mult_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                         int s, int K,
                                         double sigma, std::vector<double> a2s, std::string approach,
                                         Rcpp::List models,
                                         int nstart);

// Fit logistic regression with GLGP
// [[Rcpp::export(fit_gl_logit_gp_cpp)]]
Rcpp::List fit_gl_logit_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                               int K, Rcpp::NumericVector N_train,
                               double sigma, std::vector<double> a2s,
                               double threshold, bool sparse,
                               std::string approach,
                               Rcpp::List models,
                               bool output_cov);

// Fit logistic multinomial regression with GLGP
// [[Rcpp::export(fit_gl_logit_mult_gp_cpp)]]
Rcpp::List fit_gl_logit_mult_gp_cpp(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                    int K,
                                    double sigma, std::vector<double> a2s,
                                    double threshold, bool sparse,
                                    std::string approach,
                                    Rcpp::List models);

#endif
