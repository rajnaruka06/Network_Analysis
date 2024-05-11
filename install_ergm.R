# Path to the directory where you want to install the package
r_packages_dir <- "./r_packages"

# Check if the directory exists, if not, create it
if (!dir.exists(r_packages_dir)) {
  dir.create(r_packages_dir)
}

cat("Installing ergm package...")
# install.packages("ergm", lib=r_packages_dir, repos="https://cloud.r-project.org/")
install.packages("robustbase", lib=r_packages_dir, repos="https://cran.r-project.org/")
install.packages("coda", lib=r_packages_dir, repos="https://cran.r-project.org/")
install.packages("lpSolveAPI", lib=r_packages_dir, repos="https://cran.r-project.org/")
install.packages("trust", lib=r_packages_dir, repos="https://cran.r-project.org/")
install.packages("rle", lib=r_packages_dir, repos="https://cran.r-project.org/")

# install.packages("lifecycle", lib=r_packages_dir, repos="https://cran.r-project.org/")
# install.packages("rlang", lib=r_packages_dir, repos="https://cran.r-project.org/")
# install.packages("vctrs", lib=r_packages_dir, repos="https://cran.r-project.org/")
# install.packages("pillar", lib=r_packages_dir, repos="https://cran.r-project.org/")

install.packages("purrr", lib=r_packages_dir, repos="https://cran.r-project.org/")
install.packages("memoise", lib=r_packages_dir, repos="https://cran.r-project.org/")

install.packages("tibble", lib=r_packages_dir, repos="https://cran.r-project.org/")
install.packages("magrittr", lib=r_packages_dir, repos="https://cran.r-project.org/")
install.packages("./statnet.common_4.6.0.tar.gz", lib=r_packages_dir, repos=NULL, type="source")
install.packages("./network_1.17.0.tar.gz", lib=r_packages_dir, repos=NULL, type="source")
install.packages("./ergm_4.1.2.tar.gz", lib=r_packages_dir, repos=NULL, type="source")
install.packages("./xergm_1.8.3.tar.gz", lib=r_packages_dir, repos=NULL, type="source")
cat("\nSuccessfully installed ergm!\n")