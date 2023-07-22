// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/FLAG.h"
#include <RcppEigen.h>
#include <Rcpp.h>
#include <string>
#include <set>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// test_pgbinary_cpp
Rcpp::List test_pgbinary_cpp(const Eigen::MatrixXd& C, const Eigen::VectorXi& Y, const Eigen::MatrixXd& Cnv, int N_sample, bool output_pi);
static SEXP _FLAG_test_pgbinary_cpp_try(SEXP CSEXP, SEXP YSEXP, SEXP CnvSEXP, SEXP N_sampleSEXP, SEXP output_piSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type C(CSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXi& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Cnv(CnvSEXP);
    Rcpp::traits::input_parameter< int >::type N_sample(N_sampleSEXP);
    Rcpp::traits::input_parameter< bool >::type output_pi(output_piSEXP);
    rcpp_result_gen = Rcpp::wrap(test_pgbinary_cpp(C, Y, Cnv, N_sample, output_pi));
    return rcpp_result_gen;
END_RCPP_RETURN_ERROR
}
RcppExport SEXP _FLAG_test_pgbinary_cpp(SEXP CSEXP, SEXP YSEXP, SEXP CnvSEXP, SEXP N_sampleSEXP, SEXP output_piSEXP) {
    SEXP rcpp_result_gen;
    {
        Rcpp::RNGScope rcpp_rngScope_gen;
        rcpp_result_gen = PROTECT(_FLAG_test_pgbinary_cpp_try(CSEXP, YSEXP, CnvSEXP, N_sampleSEXP, output_piSEXP));
    }
    Rboolean rcpp_isInterrupt_gen = Rf_inherits(rcpp_result_gen, "interrupted-error");
    if (rcpp_isInterrupt_gen) {
        UNPROTECT(1);
        Rf_onintr();
    }
    bool rcpp_isLongjump_gen = Rcpp::internal::isLongjumpSentinel(rcpp_result_gen);
    if (rcpp_isLongjump_gen) {
        Rcpp::internal::resumeJump(rcpp_result_gen);
    }
    Rboolean rcpp_isError_gen = Rf_inherits(rcpp_result_gen, "try-error");
    if (rcpp_isError_gen) {
        SEXP rcpp_msgSEXP_gen = Rf_asChar(rcpp_result_gen);
        UNPROTECT(1);
        Rf_error(CHAR(rcpp_msgSEXP_gen));
    }
    UNPROTECT(1);
    return rcpp_result_gen;
}
// which_minn_rcpp
Rcpp::IntegerVector which_minn_rcpp(const Rcpp::NumericVector& z, int r);
RcppExport SEXP _FLAG_which_minn_rcpp(SEXP zSEXP, SEXP rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type z(zSEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    rcpp_result_gen = Rcpp::wrap(which_minn_rcpp(z, r));
    return rcpp_result_gen;
END_RCPP
}

// validate (ensure exported C++ functions exist before calling them)
static int _FLAG_RcppExport_validate(const char* sig) { 
    static std::set<std::string> signatures;
    if (signatures.empty()) {
        signatures.insert("Rcpp::List(*test_pgbinary_cpp)(const Eigen::MatrixXd&,const Eigen::VectorXi&,const Eigen::MatrixXd&,int,bool)");
    }
    return signatures.find(sig) != signatures.end();
}

// registerCCallable (register entry points for exported C++ functions)
RcppExport SEXP _FLAG_RcppExport_registerCCallable() { 
    R_RegisterCCallable("FLAG", "_FLAG_test_pgbinary_cpp", (DL_FUNC)_FLAG_test_pgbinary_cpp_try);
    R_RegisterCCallable("FLAG", "_FLAG_RcppExport_validate", (DL_FUNC)_FLAG_RcppExport_validate);
    return R_NilValue;
}

static const R_CallMethodDef CallEntries[] = {
    {"_FLAG_test_pgbinary_cpp", (DL_FUNC) &_FLAG_test_pgbinary_cpp, 5},
    {"_FLAG_which_minn_rcpp", (DL_FUNC) &_FLAG_which_minn_rcpp, 2},
    {"_FLAG_RcppExport_registerCCallable", (DL_FUNC) &_FLAG_RcppExport_registerCCallable, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_FLAG(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
