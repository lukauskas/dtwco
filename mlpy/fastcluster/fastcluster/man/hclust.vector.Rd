\name{hclust.vector}
\alias{hclust.vector}
\title{Fast hierarchical, agglomerative clustering of vector data}
\description{
This function implements hierarchical, agglomerative clustering with memory-saving algorithms.}
\usage{hclust.vector(X, method="single", members=NULL, metric='euclidean', p=NULL)}
\arguments{
\item{X}{an \eqn{(N\times D)}{(N×D)} matrix of '\link{double}' values:
  \eqn{N}{N} observations in \eqn{N}{N} variables.}
\item{method}{the agglomeration method to be used. This must be (an
  unambiguous abbreviation of) one of \code{"single"},
  \code{"ward"}, \code{"centroid"} or \code{"median"}.}
\item{members}{\code{NULL} or a vector with length the number of observations.}
\item{metric}{the distance measure to be used. This must be one of
  \code{"euclidean"}, \code{"maximum"}, \code{"manhattan"},
  \code{"canberra"}, \code{"binary"} or \code{"minkowski"}. Any
  unambiguous substring can be given.}
\item{p}{parameter for the Minkowski metric.}
}
\details{The function \code{\link{hclust.vector}} provides clustering when the
  input is vector data. It uses memory-saving algorithms which allow
  processing of larger data sets than \code{\link{hclust}} does.

  The \code{"ward"}, \code{"centroid"} and \code{"median"} methods
  require \code{metric="euclidean"} and cluster the data set with
  respect to Euclidean distances.

  For \code{"single"} linkage clustering, any dissimilarity
  measure may be chosen. Currently, the same metrics are implemented as the
  \code{\link[stats:dist]{dist}} function provides.

  The call\preformatted{  hclust.vector(X, method='single', metric=[...])}
  gives the same result as\preformatted{  hclust(dist(X, metric=[...]), method='single')}
  but uses less memory and is equally fast.

  For the Euclidean methods, care must be taken since
  \code{\link{hclust}} expects \bold{squared} Euclidean
  distances. Hence, the call\preformatted{  hclust.vector(X, method='ward')}
  is, aside from the lesser memory requirements, equivalent to\preformatted{  d = dist(X)
  hc = hclust(d^2, method='ward')
  hc$height = sqrt(hc$height)}
  The same applies to the \code{"centroid"} and \code{"median"} methods.

  More details are in the User's manual
  \href{http://cran.r-project.org/web/packages/fastcluster/vignettes/fastcluster.pdf}{fastcluster.pdf}, which is available as
  a vignette. Get this from the R command line with
  \code{vignette('fastcluster')}.
}
\references{\url{http://math.stanford.edu/~muellner}}
\author{Daniel Müllner}
\seealso{\code{\link{fastcluster}}, \code{\link{hclust}}}
\examples{# Taken and modified from stats::hclust
## Perform centroid clustering with squared Euclidean distances,
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
