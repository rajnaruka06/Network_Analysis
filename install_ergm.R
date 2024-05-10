# Path to the directory where you want to install the package
r_packages_dir <- "./r_packages"

# Check if the directory exists, if not, create it
if (!dir.exists(r_packages_dir)) {
  dir.create(r_packages_dir)
}

# Install ergm package to the specified directory
install.packages("ergm", lib=r_packages_dir)