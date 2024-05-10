/*  File src/init.c in package ergm, part of the
 *  Statnet suite of packages for network analysis, https://statnet.org .
 *
 *  This software is distributed under the GPL-3 license.  It is free,
 *  open source, and has the attribution requirements (GPL Section 7) at
 *  https://statnet.org/attribution .
 *
 *  Copyright 2003-2023 Statnet Commons
 */
/* This code was procedurally generated by running

   > tools::package_native_routine_registration_skeleton(".", "./src/init.c")

   in R started in the package's root directory, then changing

   "R_useDynamicSymbols(dll, FALSE)" to "R_useDynamicSymbols(dll, TRUE)".
*/

#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* .C calls */
extern void full_geodesic_distribution(void *, void *, void *, void *, void *, void *, void *, void *);

/* .Call calls */
extern SEXP AllStatistics(SEXP, SEXP);
extern SEXP CD_wrapper(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP ergm_eta_wrapper(SEXP, SEXP);
extern SEXP ergm_etagrad_wrapper(SEXP, SEXP);
extern SEXP ergm_etagradmult_wrapper(SEXP, SEXP, SEXP);
extern SEXP ErgmStateArrayClear(void);
extern SEXP ErgmWtStateArrayClear(void);
extern SEXP get_ergm_omp_terms(void);
extern SEXP Godfather_wrapper(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP MCMC_wrapper(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP MCMCPhase12(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP MPLE_workspace_free(void);
extern SEXP MPLE_wrapper(SEXP, SEXP, SEXP);
extern SEXP network_stats_wrapper(SEXP);
extern SEXP SAN_wrapper(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP set_ergm_omp_terms(SEXP);
extern SEXP wt_network_stats_wrapper(SEXP);
extern SEXP WtCD_wrapper(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP WtGodfather_wrapper(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP WtMCMC_wrapper(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP WtMCMCPhase12(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP WtSAN_wrapper(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CMethodDef CEntries[] = {
    {"full_geodesic_distribution", (DL_FUNC) &full_geodesic_distribution, 8},
    {NULL, NULL, 0}
};

static const R_CallMethodDef CallEntries[] = {
    {"AllStatistics",            (DL_FUNC) &AllStatistics,             2},
    {"CD_wrapper",               (DL_FUNC) &CD_wrapper,                5},
    {"ergm_eta_wrapper",         (DL_FUNC) &ergm_eta_wrapper,          2},
    {"ergm_etagrad_wrapper",     (DL_FUNC) &ergm_etagrad_wrapper,      2},
    {"ergm_etagradmult_wrapper", (DL_FUNC) &ergm_etagradmult_wrapper,  3},
    {"ErgmStateArrayClear",      (DL_FUNC) &ErgmStateArrayClear,       0},
    {"ErgmWtStateArrayClear",    (DL_FUNC) &ErgmWtStateArrayClear,     0},
    {"get_ergm_omp_terms",       (DL_FUNC) &get_ergm_omp_terms,        0},
    {"Godfather_wrapper",        (DL_FUNC) &Godfather_wrapper,         6},
    {"MCMC_wrapper",             (DL_FUNC) &MCMC_wrapper,              7},
    {"MCMCPhase12",              (DL_FUNC) &MCMCPhase12,              11},
    {"MPLE_workspace_free",      (DL_FUNC) &MPLE_workspace_free,       0},
    {"MPLE_wrapper",             (DL_FUNC) &MPLE_wrapper,              3},
    {"network_stats_wrapper",    (DL_FUNC) &network_stats_wrapper,     1},
    {"SAN_wrapper",              (DL_FUNC) &SAN_wrapper,               9},
    {"set_ergm_omp_terms",       (DL_FUNC) &set_ergm_omp_terms,        1},
    {"wt_network_stats_wrapper", (DL_FUNC) &wt_network_stats_wrapper,  1},
    {"WtCD_wrapper",             (DL_FUNC) &WtCD_wrapper,              5},
    {"WtGodfather_wrapper",      (DL_FUNC) &WtGodfather_wrapper,       6},
    {"WtMCMC_wrapper",           (DL_FUNC) &WtMCMC_wrapper,            7},
    {"WtMCMCPhase12",            (DL_FUNC) &WtMCMCPhase12,            11},
    {"WtSAN_wrapper",            (DL_FUNC) &WtSAN_wrapper,             9},
    {NULL, NULL, 0}
};

void R_init_ergm(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, TRUE);
}