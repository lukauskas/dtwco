#  fastcluster: Fast hierarchical clustering routines for R and Python
#
#  Copyright © 2011 Daniel Müllner
#  <http://math.stanford.edu/~muellner>
#
# Test script for the R interface

seed = as.integer(runif(1, 0, 1e9))
set.seed(seed)
cat(sprintf("Random seed: %d\n",seed))

print_seed <- function() {
  return(sprintf('
Please send a report to the author of the \'fastcluster\' package, Daniel Müllner.
For contact details, see <http://math.stanford.edu/~muellner>. To make the error
reproducible, you must include the following number (the random seed value) in
your error report: %d.\n\n', seed))
}

# Compare two dendrograms and check whether they are equal, except that
# ties may be resolved differently.
compare <- function(dg1, dg2) {
  h1 <- dg1$height
  h2 <- dg2$height
  # "height" vectors may have small numerical errors.
  rdiffs <- abs(h1-h2)/pmax(abs(h1),abs(h2))
  rdiffs = rdiffs[complete.cases(rdiffs)]
  rel_error <-  max(rdiffs)
  # We allow a relative error of 1e-13.
  if (rel_error>1e-13) {
    print(h1)
    print(h2)
    cat(sprintf('Height vectors differ! The maximum relative error is %e.\n', rel_error))
    return(FALSE)
  }
  # Filter the indices where consecutive merging distances are distinct.
  d = diff(dg1$height)
  b = (c(d,1)!=0 & c(1,d)!=0)
  #cat(sprintf("Percentage of indices where we can test: %g.\n",100.0*length(b[b])/length(b)))
  if (any(b)) {
    m1 = dg1$merge[b,] 
    m2 = dg2$merge[b,]

    r = function(i) {
      if (i<0) {
        return(1)
      }
      else {
        return(b[i])
      }
    }

    f = sapply(m1,r)
    fm1 = m1*f
    fm2 = m2*f
    # The "merge" matrices must be identical whereever indices are not ambiguous
    # due to ties.
    if (!identical(fm1,fm2)) {
      cat('Merge matrices differ!\n')
      return(FALSE)
    }
    # Compare the "order" vectors only if all merging distances were distinct.
    if (all(b) && !identical(dg1$order,dg2$order)) {
      cat('Order vectors differ!\n')
      return(FALSE)
    }
  }
  return(TRUE)
}

# Generate uniformly distributed random data
generate.uniform <- function() {
  n = sample(10:1000,1)
  range_exp = runif(1,min=-10, max=10)
  cat(sprintf("Number of sample points: %d\n",n))
  cat(sprintf("Dissimilarity range: [0,%g]\n",10^range_exp))
  d = runif(n*(n-1)/2, min=0, max=10^range_exp)
  # Fake a compressed distance matrix
  attributes(d) <- NULL
  attr(d,"Size") <- n
  attr(d, "call") <- 'N/A'
  class(d) <- "dist"
  return(d)
}

# Generate normally distributed random data
generate.normal <- function() {
  n = sample(10:1000,1)
  dim = sample(2:20,1)

  cat (sprintf("Number of sample points: %d\n",n))
  cat (sprintf("Dimension: %d\n",dim))

  pcd = matrix(rnorm(n*dim), c(n,dim))
  d = dist(pcd)
  return(d)
}

# Test the clustering functions when a distance matrix is given.
test.dm <-  function(d) {
  d2 = d
  for (method in c('single','complete','average','mcquitty','ward','centroid','median') ) {
    cat(paste('Method :', method, '\n'))
    dg_fastcluster = fastcluster::hclust(d, method=method)
    dg_stats       = stats::hclust(d, method=method)
    if (!identical(d,d2)) {
      cat('Input array was corrupted!\n')
      stop(print_seed())
    }
    if (!compare(dg_stats, dg_fastcluster)) {
      stop(print_seed())
    }
  }
  cat('Passed.\n')
}

# Test the clustering functions for vector input in Euclidean space.
test.vector <-  function() {
  # generate test data
  n = sample(10:1000,1)
  dim = sample(2:20,1)
  cat (sprintf("Number of sample points: %d\n",n))
  cat (sprintf("Dimension: %d\n",dim))

  range_exp = runif(1,min=-10, max=10)
  pcd = matrix(rnorm(n*dim, sd=10^range_exp), c(n,dim))
  pcd2 = pcd
  # test
  method='single'
  cat(paste('Method:', method, '\n'))
  for (metric in c('euclidean', 'maximum', 'manhattan', 'canberra', 'minkowski')) {
    cat(paste('    Metric:', metric, '\n'))
    if (metric=='minkowski') {
      p = runif(1, min=1.0, max=10.0)
      cat (sprintf("    p: %g\n",p));
      dg_fastcluster = fastcluster::hclust.vector(pcd, method=method, metric=metric, p=p)
      d = dist(pcd, method=metric, p=p)
    }
    else {
      dg_fastcluster = fastcluster::hclust.vector(pcd, method=method, metric=metric)
      d = dist(pcd, method=metric)
    }
    d2 = d
    dg_fastcluster_dist = fastcluster::hclust(d, method=method)
    if (!identical(d,d2) || !identical(pcd,pcd2)) {
      cat('Input array was corrupted!\n')
      stop(print_seed())
    }
    if (!compare(dg_fastcluster_dist, dg_fastcluster)) {
      stop(print_seed())
    }
  }
  for (method in c('ward','centroid','median') ) {
    cat(paste('Method:', method, '\n'))
    dg_fastcluster = fastcluster::hclust.vector(pcd, method=method)
    if (!identical(pcd,pcd2)) {
      cat('Input array was corrupted!\n')
      stop(print_seed())
    }
    d = dist(pcd)
    # Workaround: fastcluster::hclust expects _squared_ euclidean distances.
    d = d^2
    d2 = d
    dg_fastcluster_dist = fastcluster::hclust(d, method=method)
    if (!identical(d,d2)) {
      cat('Input array was corrupted!\n')
      stop(print_seed())
    }
    dg_fastcluster_dist$height = sqrt(dg_fastcluster_dist$height)
    # The Euclidean methods may have small numerical errors due to squaring/
    # taking the root in the Euclidean distances.
    if (!compare(dg_fastcluster_dist, dg_fastcluster)) {
      stop(print_seed())
    }
  }
  cat('Passed.\n')
}

# Test the single linkage function with the "binary" metric 
test.vector.binary <- function() {
  # generate test data
  cat (sprintf("Uniform sampling for the 'binary' metric:\n"))
  n = sample(10:400,1)
  dim = sample(n:(2*n),1)
  cat (sprintf("Number of sample points: %d\n",n))
  cat (sprintf("Dimension: %d\n",dim))
  pcd = matrix(sample(-1:2, n*dim, replace=T), c(n,dim))
  pcd2 = pcd
  # test
  method='single'
  metric='binary'
  cat(paste('Method:', method, '\n'))
  cat(paste('    Metric:', metric, '\n'))
  dg_fastcluster = fastcluster::hclust.vector(pcd, method=method, metric=metric)
  d = dist(pcd, method=metric)
  d2 = d
  dg_fastcluster_dist       = fastcluster::hclust(d, method=method)
  if (!identical(d,d2) || !identical(d,d2)) {
    cat('Input array was corrupted!\n')
    stop(print_seed())
  }
  if (!compare(dg_fastcluster_dist, dg_fastcluster)) {
    stop(print_seed())
  }
  cat('Passed.\n')
}


N = 25
for (i in (1:N)) {
  if (i%%2==1) {
    cat(sprintf('Random test %d of %d (uniform distribution of distances):\n',i,2*N))
    d = generate.uniform()
  }
  else {
    cat(sprintf('Random test %d of %d (Gaussian density):\n',i,2*N))
    d = generate.normal()
  }
  test.dm(d)
}
for (i in (N+1:N)) {
  cat(sprintf('Random test %d of %d (Gaussian density):\n',i,2*N))
  test.vector()
  test.vector.binary()
}

cat('Done.\n')
