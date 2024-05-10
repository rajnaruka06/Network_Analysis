# Path to the directory where you want to install the package
r_packages_dir <- "./r_packages"

# Check if the directory exists, if not, create it
if (!dir.exists(r_packages_dir)) {
  dir.create(r_packages_dir)
}

cat("Installing ergm package...")
install.packages("./ergm_4.6.0.tar.gz", lib=r_packages_dir, repos=NULL, type="source")
cat("\nSuccessfully installed ergm!\n")