\name{hclust}
\alias{hclust}
\title{Fast hierarchical, agglomerative clustering of dissimilarity data}
\description{
This function implements hierarchical clustering with the same interface as \code{\link[stats:hclust]{hclust}} from the \pkg{\link{stats}} package but with much faster algorithms.
}
\usage{hclust(d, method="complete", members=NULL)}
\arguments{
\item{d}{a dissimilarity structure as produced by \code{dist}.}
\item{method}{the agglomeration method to be used. This must be (an
  unambiguous abbreviation of) one of \code{"single"},
  \code{"complete"}, \code{"average"}, \code{"mcquitty"},
  \code{"ward"}, \code{"centroid"} or \code{"median"}.}
\item{members}{\code{NULL} or a vector with length the number of
  observations.}
}
\value{An object of class \code{'hclust'}. It encodes a stepwise dendrogram.}
\details{See the documentation of the original function
  \code{\link[stats:hclust]{hclust}} in the \pkg{\link{stats}} package.

   A comprehensive User's manual
    \href{http://cran.r-project.org/web/packages/fastcluster/vignettes/fastcluster.pdf}{fastcluster.pdf} is available as a vignette. Get this from the R command line with \code{vignette('fastcluster')}.
}
\references{\url{http://math.stanford.edu/~muellner}}
\author{Daniel Müllner}
\seealso{\code{\link{fastcluster}}, \code{\link{hclust.vector}}, \code{\link[stats:hclust]{stats::hclust}}} 
\examples{# Taken and modified from stats::hclust
#
# hclust(...)        # new method
# stats::hclust(...) # old method

require(fastcluster)
require(graphics)

hc <- hclust(dist(USArrests), "ave")
plot(hc)
plot(hc, hang = -1)

## Do the same with centroid clustering and squared Euclidean distance,
## cut the tree into ten clusters and reconstruct the upper part of the
## tree from the cluster centers.
hc <- hclust(dist(USArrests)^2, "cen")
memb <- cutree(hc, k = 10)
cent <- NULL
for(k in 1:10){
  cent <- rbind(cent, colMeans(USArrests[memb == k, , drop = FALSE]))
}
hc1 <- hclust(dist(cent)^2, method = "cen", members = table(memb))
opar <- par(mfrow = c(1, 2))
plot(hc,  labels = FALSE, hang = -1, main = "Original Tree")
plot(hc1, labels = FALSE, hang = -1, main = "Re-start from 10 clusters")
par(opar)
}
\keyword{multivariate}
\keyword{cluster}
