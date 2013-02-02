\name{fastcluster}
\alias{fastcluster}
\alias{fastcluster-package}
\docType{package}
\title{Fast hierarchical, agglomerative clustering routines for R and Python}
\description{The \pkg{fastcluster} package provides efficient algorithms for hierarchical,
agglomerative clustering. In addition to the R interface, there is also
a Python interface to the underlying C++ library, to be found in the
source distribution.
}
\details{The function \code{\link{hclust}} provides clustering when the
  input is a dissimilarity matrix. A dissimilarity matrix can be
  computed from vector data by \code{\link{dist}}. The
  \code{\link{hclust}} function can be used as a drop-in replacement for
  existing routines: \code{\link[stats:hclust]{stats::hclust}} and
  \code{\link[flashClust:hclust]{flashClust::hclust}} alias
  \code{\link[flashClust:flashClust]{flashClust::flashClust}}. Once the
  fastcluster library is loaded at the beginning of the code, every
  program that uses hierarchical clustering can benefit immediately and
  effortlessly from the performance gain

  When the package is loaded, it overwrites the function
  \code{\link{hclust}} with the new code.

  The function \code{\link{hclust.vector}} provides memory-saving routines
  when the input is vector data.

  Further information:
  \itemize{
    \item R documentation pages: \code{\link{hclust}},
    \code{\link{hclust.vector}}

    \item A comprehensive User's manual:
    \href{http://cran.r-project.org/web/packages/fastcluster/vignettes/fastcluster.pdf}{fastcluster.pdf}. Get this from the R
    command line with \code{vignette('fastcluster')}.

    \item See the author's home page for a performance comparison:
  \url{http://math.stanford.edu/~muellner/fastcluster.html}.
    }
}
\references{\url{http://math.stanford.edu/~muellner}}
\author{Daniel Müllner}
\seealso{\code{\link{hclust}}, \code{\link{hclust.vector}}}
\examples{# Taken and modified from stats::hclust
#
# hclust(...)        # new method
# hclust.vector(...) # new method
# stats::hclust(...) # old method

require(fastcluster)
require(graphics)

hc <- hclust(dist(USArrests), "ave")
plot(hc)
plot(hc, hang = -1)

## Do the same with centroid clustering and squared Euclidean distance,
## cut the tree into ten clusters and reconstruct the upper part of the
## tree from the cluster centers.
hc <- hclust.vector(USArrests, "cen")
# squared Euclidean distances
hc$height <- hc$height^2
memb <- cutree(hc, k = 10)
cent <- NULL
for(k in 1:10){
  cent <- rbind(cent, colMeans(USArrests[memb == k, , drop = FALSE]))
}
hc1 <- hclust.vector(cent, method = "cen", members = table(memb))
# squared Euclidean distances
hc1$height <- hc1$height^2
opar <- par(mfrow = c(1, 2))
plot(hc,  labels = FALSE, hang = -1, main = "Original Tree")
plot(hc1, labels = FALSE, hang = -1, main = "Re-start from 10 clusters")
par(opar)
}
\keyword{multivariate}
\keyword{cluster}
