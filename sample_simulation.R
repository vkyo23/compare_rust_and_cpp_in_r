# 0. setup -----
## initialize workspace
rm(list = ls())
set.seed(1)

## install packages
### CppIRT
#remotes::install_github("vkyo23/compare_rust_and_cpp_in_r", subdir = "CppIRT")

### RustIRT
#remotes::install_github("vkyo23/compare_rust_and_cpp_in_r", subdir = "RustIRT")


# 1. data generating process -----
dgp <- function(I, J) {
  alpha <- rnorm(J)
  beta <- rnorm(J)
  theta <- rnorm(I)

  p <- plogis(
    cbind(1, theta) %*% rbind(alpha, beta)
  )
  Y <- rbinom(I * J, 1, p) |> 
    matrix(I, J)

  out <- list(
    alpha = alpha,
    beta = beta,
    theta = theta,
    Y = Y
  )

  return(out)
}

I <- 1000
J <- 100
data <- dgp(I = I, J = J)


# 2. Experiment I: Simple comparison -----
## 2.1 CppIRT -----
model_cpp <- CppIRT::cppirt_model(
  data = data$Y,
  control = list(
    theta_constraint = which.max(data$theta),
    theta_strict_identification = TRUE
  )
)

stime <- proc.time()[3]
model_cpp$fit()
etime <- proc.time()[3]
cat("CppIRT:", round(etime - stime, 3), "sec.\n")

out_cpp <- model_cpp$output()


## 2.2 RustIRT -----
model_rust <- RustIRT::rustirt_model(
  data = data$Y,
  control = list(
    theta_constraint = which.max(data$theta),
    theta_strict_identification = TRUE
  )
)

stime <- proc.time()[3]
model_rust$fit()
etime <- proc.time()[3]
cat("RustIRT:", round(etime - stime, 3), "sec.\n")

out_rust <- model_rust$output()


## 2.3 Result -----
### 2.3.1. true vs. estimated -----
df <- dplyr::tibble(
  label = c(
    rep("alpha", J),
    rep("beta", J),
    rep("theta", I)
  ),
  true = c(data$alpha, data$beta, data$theta),
  estimated = c(out_rust$alpha, out_rust$beta, out_rust$theta)
)
corr_df <- df |> 
  dplyr::summarise(
    corr = paste("cor =", round(stats::cor(true, estimated), 3)),
    .by = label
  )

png("figures/corr_rust_true.png", width = 1600, height = 900, res = 200)
ggplot2::ggplot() +
  ggplot2::geom_abline(
    slope = 1, intercept = 0, alpha = .2
  ) +
  ggplot2::geom_point(
    data = df, 
    ggplot2::aes(x = estimated, y = true),
    alpha = .4
  ) +
  ggplot2::facet_wrap(~ label) +
  ggplot2::geom_text(
    data = corr_df, 
    ggplot2::aes(label = corr),
    x = 2,
    y = -2
  ) +
  ggplot2::theme_light() +
  ggplot2::xlab("Estimated values") +
  ggplot2::ylab("True")
dev.off()

### 2.3.2. cpp vs. rust -----
df <- dplyr::tibble(
  label = c(
    rep("alpha", J),
    rep("beta", J),
    rep("theta", I)
  ),
  cpp = c(out_cpp$alpha, out_cpp$beta, out_cpp$theta),
  rust = c(out_rust$alpha, out_rust$beta, out_rust$theta)
)

#### Correlation
corr_df <- df |> 
  dplyr::summarise(
    corr = paste("cor =", round(stats::cor(cpp, rust), 3)),
    .by = label
  )

png("figures/corr_rust_cpp.png", width = 1600, height = 900, res = 200)
ggplot2::ggplot() +
  ggplot2::geom_abline(
    slope = 1, intercept = 0, alpha = .2
  ) +
  ggplot2::geom_point(
    data = df, 
    ggplot2::aes(x = rust, y = cpp),
    alpha = .4
  ) +
  ggplot2::facet_wrap(~ label) +
  ggplot2::geom_text(
    data = corr_df, 
    ggplot2::aes(label = corr),
    x = 2,
    y = -2
  ) +
  ggplot2::theme_light() +
  ggplot2::xlab("Rust estimates") +
  ggplot2::ylab("Cpp estimates")
dev.off()

#### RMSE, MAE
df |> 
  dplyr::mutate(
    sq_err = (cpp - rust)^2,
    ae = abs(cpp - rust)
  ) |> 
  dplyr::summarise(
    RMSE = sqrt(mean(sq_err)),
    MAE = mean(ae),
    .by = label
  ) 


# 3. Experiment II: Time performance with different sample sizes ----
Is <- seq(1000, 20000, 500)
J <- 100

result_e2 <- matrix(NA, length(Is), 2) |> 
  dplyr::as_tibble() |>
  dplyr::rename(Cpp = V1, Rust = V2) |>
  dplyr::mutate(
    i_size = Is,
    .before = Cpp
  )

for (i in seq_along(Is)) {
  data <- dgp(I = Is[i], J = J)

  model_cpp <- CppIRT::cppirt_model(
    data = data$Y,
    control = list(
      theta_constraint = which.max(data$theta),
      theta_strict_identification = TRUE
    )
  )

  stime <- proc.time()[3]
  model_cpp$fit()
  result_e2[i, 2] <- round(proc.time()[3] - stime, 3)

  model_rust <- RustIRT::rustirt_model(
    data = data$Y,
    control = list(
      theta_constraint = which.max(data$theta),
      theta_strict_identification = TRUE
    )
  )

  stime <- proc.time()[3]
  model_rust$fit()
  result_e2[i, 3] <- round(proc.time()[3] - stime, 3)
}

png("figures/estimation_time.png", width = 1600, height = 900, res = 200)
result_e2 |> 
  tidyr::pivot_longer(-i_size) |> 
  ggplot2::ggplot(ggplot2::aes(x = i_size, y = value, color = name)) +
  ggplot2::geom_point(ggplot2::aes(shape = name)) +
  ggplot2::geom_line(ggplot2::aes(linetype = name)) +
  ggplot2::scale_color_grey(start = 0, end = 0.5) +
  ggplot2::xlab("# of i") +
  ggplot2::ylab("Time (sec.)") +
  ggplot2::theme_light() +
  ggplot2::theme(legend.title = ggplot2::element_blank())
dev.off()
