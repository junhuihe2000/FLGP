#include <Rcpp.h>
using namespace Rcpp;

//' Find indexes of the first r smallest elements in arrays based on Bubblesorting
//'
//' @param z Numeric vector, including elements to be sorted.
//' @param r Int, the number of the smallest elements.
//'
//' @return Integer vector of length r, indicating the indexes of the first r smallest elements.
//' @export
//'
//' @examples
//' z <- c(1,3,2)
//' r <- 2
//' which_minn_rcpp(z, r)
// [[Rcpp::export(which_minn_rcpp)]]
Rcpp::IntegerVector which_minn_rcpp(const Rcpp::NumericVector& z, int r) {
  int n = z.length();
  Rcpp::IntegerVector ind_z = Rcpp::seq_len(n);
  int tmp;
  for(int i=0;i<r;i++) {
    for(int j=n-1;j>i;j--) {
      if(z[ind_z[j]-1]<z[ind_z[j-1]-1]) {
        tmp = ind_z[j]; ind_z[j] = ind_z[j-1]; ind_z[j-1] = tmp;
      }
    }
  }
  return Rcpp::head(ind_z, r);
}
