#' @title Fitting the two parameter logistic IRT model
#' @description \code{\link{rustirt_model}} is an instance to estimate IRT. For the methods, see \code{\link{RustIRTModel}}.
#'
#' @param data an I (individual) x J (item) matrix. This should contain 0 or 1.
#' @param priors a list containing priors:
#' \itemize{
#'   \item \code{a0}: a double. Prior mean of alpha. Default is 0.
#'   \item \code{A0}: a double (> 0). Prior variance of alpha. Default is 25.
#'   \item \code{b0}: a double. Prior mean of beta. Default is 0.
#'   \item \code{B0}: a double (> 0). Prior variance of beta. Default is 25.
#' }
#' @param control a list containing model controls:
#' \itemize{
#'   \item \code{theta_constraint}: a positive integer. The position of an individual whose \code{theta} is always set positive.
#'   \item \code{theta_strict_identification}: a bool. Whether to standardize \code{theta} for identification.
#'   \item \code{maxit}: a positive integer. Maximum number of EM iteration. Default is 500.
#'   \item \code{verbose}: a positive integer. Print the status every \code{verbose} iteration. If this is not specified, any iteration status is not be printed.
#'   \item \code{tol}: a float (0.0 < tol < 1.0). Convergence threshold. Default is 1e-6.
#' }
#'
#' @return A \code{\link{RustIRTModel}} object.
#' @export
#'
#' @examples
#' \dontrun{
#'  library(RustIRT)
#'
#'  # data generating process
#'  I <- 1000
#'  J <- 100
#'  alpha <- rnorm(J)
#'  beta <- rnorm(J)
#'  theta <- rnorm(I)
#'  p <- plogis(
#'   cbind(1, theta) %*% rbind(alpha, beta)
#'  )
#'  Y <- rbinom(I * J, 1, p) |>
#'         matrix(I, J)
#'
#'  # create an instance
#'  model <- rustirt_model(
#'    data = Y,
#'    control = list(
#'      theta_constraint = which.max(theta),
#'      theta_strict_identification = TRUE
#'    )
#'  )
#'
#'  # fit the model
#'  model$fit()
#'
#'  # get the result
#'  out <- model$output()
#' }

rustirt_model <- function(data, priors = list(), control = list()) {
  return(RustIRTModel$new(data = data, priors = priors, control = control))
}

#' RustIRTModel objects
#'
#' @name RustIRTModel
#' @description A \code{\link{RustIRTModel}} object is an \code{\link[R6]{R6Class}} object created
#'   by the \code{\link{rustirt_model}} function.
#'
#' @importFrom R6 R6Class
#' @importFrom stats coef glm
#'
#' @useDynLib RustIRT, .registration = TRUE
#'

RustIRTModel <- R6::R6Class(
  classname = "RustIRTModel",
  lock_objects = FALSE,
  public = list(

    #' @description Initialize the new instance of \code{\link{RustIRTModel}}.
    #'
    #' @param data an I (individual) x J (item) matrix. This should contain 0 or 1.
    #' @param priors a list containing priors:
    #' \itemize{
    #'   \item \code{a0}: a double. Prior mean of alpha. Default is 0.
    #'   \item \code{A0}: a double (> 0). Prior variance of alpha. Default is 25.
    #'   \item \code{b0}: a double. Prior mean of beta. Default is 0.
    #'   \item \code{B0}: a double (> 0). Prior variance of beta. Default is 25.
    #' }
    #' @param control a list containing model controls:
    #' \itemize{
    #'   \item\code{theta_constraint}: a positive integer. The position of an individual whose \code{theta} is always set positive.
    #'   \item\code{theta_strict_identification}: a bool. Whether to standardize \code{theta} for identification.
    #'   \item\code{maxit}: a positive integer. Maximum number of EM iteration. Default is 500.
    #'   \item\code{verbose}: a positive integer. Print the status every \code{verbose} iteration. If this is not specified, any iteration status is not be printed.
    #'   \item\code{tol}: a float (0.0 < tol < 1.0). Convergence threshold. Default is 1e-6.
    #' }
    initialize = function(data, priors, control) {
      self$Y <- data
      self$alpha <- self$beta <- rep(0, ncol(data))
      self$theta <- rep(0, nrow(data))
      self$a0 <- ifelse(!exists("a0", priors), 0, priors$a0)
      self$A0 <- ifelse(!exists("A0", priors), 25, priors$A0)
      self$b0 <- ifelse(!exists("b0", priors), 0, priors$b0)
      self$B0 <- ifelse(!exists("B0", priors), 25, priors$B0)
      self$theta_constraint <- ifelse(
        !exists("theta_constraint", control),
        -1,
        control$theta_constraint
      )
      self$theta_strict_identification <- ifelse(
        !exists("theta_strict_identification", control),
        FALSE,
        control$theta_strict_identification
      )
      self$maxit <- ifelse(!exists("maxit", control), 500, control$maxit)
      self$verbose <- ifelse(!exists("verbose", control), 10, control$verbose)
      self$tol <- ifelse(!exists("tol", control), 1e-6, control$tol)
    },
    #' @description Fit the model via Polya-Gamma data augmentation and EM algorithm. Note that this method is a void function.
    fit = function() {
      cat("* initialized the model...")
      private$initialize_parameters()
      cat("DONE\n")

      cat("* fitting the model:\n")
      private$modelfit <- fit_RustIRT(
        Y = self$Y,
        alpha = self$alpha,
        beta = self$beta,
        theta = self$theta,
        a0 = self$a0,
        A0 = self$A0,
        b0 = self$b0,
        B0 = self$B0,
        theta_constraint = self$theta_constraint - 1,
        theta_strict_identification = self$theta_strict_identification,
        maxit = self$maxit,
        verbose = self$verbose,
        tol = self$tol
      )

      if (private$modelfit$converged) {
        cat("* model converged at iteration", private$modelfit$iteration, "\n")
      } else {
        cat("* model failed to converge\n")
      }
    },
    #' @description Return the model fitting result.
    #' @return A list object.
    output = function() {
      modellist <- list(
        data = self$Y,
        priors <- list(
          a0 = self$a0,
          A0 = self$A0,
          b0 = self$b0,
          B0 = self$B0
        ),
        control = list(
          theta_constraint = self$theta_constraint,
          theta_strict_identification = self$theta_strict_identification,
          maxit = self$maxit,
          verbose = self$verbose,
          tol = self$tol
        )
      )
      private$modelfit$model <- modellist
      return(private$modelfit)
    }
  ),
  private = list(
    modelfit = list(),
    initialize_parameters = function() {
      self$theta <- stats::prcomp(self$Y)$x[, 1]
      if (self$theta_constraint == -1) self$theta_constraint <- which.max(self$theta)
      if (self$theta[self$theta_constraint] < 0) self$theta <- -self$theta
      self$theta <- scale(self$theta)[, 1]

      for (j in 1:ncol(self$Y)) {
        coefs <- stats::glm(self$Y[, j] ~ self$theta, family = "binomial") |>
          stats::coef()
        coefs <- ifelse(abs(coefs) >= 10, 0.1, coefs)
        self$alpha[j] <- coefs[1]
        self$beta[j] <- coefs[2]
      }
    }
  )
)
