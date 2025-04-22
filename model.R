library(tidyverse)
library(plotly)
library(randomForest)
library(dplyr)
library(caret)
library(yardstick)
library(MASS)
library(mgcv)
library(knitr)
library(kableExtra)
library(xgboost)

df <- read.csv("data/merged_data.csv")

df <- df |>
  distinct() |>
  drop_na() |> 
  mutate(
    date = as.Date(date),
    hour = as.factor(hour),
    weather_main = as.factor(weather_main),
    weather_desc = as.factor(weather_desc)
  )

set.seed(123)
split_index <- createDataPartition(df$total_trips, p = 0.8, list = FALSE)
train_df <- df[split_index, ]
test_df  <- df[-split_index, ]

# Linear Regression Model
lg_train <- train_df
lg_test <- test_df

lg_train <- lg_train %>%
  mutate(
    total_trips_bc    = total_trips^0.5,
    humidity_bc       = humidity^2,
    wind_speed_bc     = wind_speed^0.25,
    cloudiness_bc     = cloudiness^0.5
  )

lg_test <- lg_test %>%
  mutate(
    total_trips_bc    = total_trips^0.5,
    humidity_bc       = humidity^2,
    wind_speed_bc     = wind_speed^0.25,
    cloudiness_bc     = cloudiness^0.5
  )

final_lm <- lm(total_trips_bc ~ temp + pressure + humidity_bc + wind_speed_bc +
                 cloudiness_bc + hour + weather_main,
               data = lg_train)

lg_train$pred_bc <- predict(final_lm, newdata = lg_train)
lg_test$pred_bc  <- predict(final_lm, newdata = lg_test)

lg_train$predicted <- (lg_train$pred_bc)^2
lg_test$predicted  <- (lg_test$pred_bc)^2

metrics_train_lm <- metrics(lg_train, truth = total_trips, estimate = predicted)
metrics_test_lm  <- metrics(lg_test,  truth = total_trips, estimate = predicted)

# Generalized Linear Model (GLM)
glm_train <- train_df
glm_test <- test_df

nb_model <- MASS::glm.nb(
  total_trips ~ temp + humidity + wind_speed + hour + weather_main,
  data = glm_train
)
final_glm <- MASS::stepAIC(nb_model, direction = "both", trace = FALSE)

glm_train$predicted <- predict(final_glm, type = "response")
glm_test$predicted  <- predict(final_glm, newdata = glm_test, type = "response")


metrics_train_glm <- glm_train |>
  yardstick::metrics(truth = total_trips, estimate = predicted)

metrics_test_glm <- glm_test |>
  yardstick::metrics(truth = total_trips, estimate = predicted)

# Generalized Additive Model (GAM)
gam_train <- train_df
gam_test  <- test_df

final_gam <- gam(
  total_trips ~ s(temp) + s(wind_speed) +
    hour + weather_main,
  data = gam_train,
  family = nb(link = "log")
)

gam_train$predicted <- predict(final_gam, type = "response")
gam_test$predicted  <- predict(final_gam, newdata = gam_test, type = "response")

metrics_train_gam <- metrics(gam_train, truth = total_trips, estimate = predicted)
metrics_test_gam  <- metrics(gam_test,  truth = total_trips, estimate = predicted)

df_temp <- data.frame(
  temp = c(-0.43000000, -0.09616162, 0.23767677, 0.57151515, 0.90535354, 1.23919192, 1.57303030, 1.90686869, 2.24070707,
           2.57454545, 2.90838384, 3.24222222, 3.57606061, 3.90989899, 4.24373737, 4.57757576, 4.91141414, 5.24525253,
           5.57909091, 5.91292929, 6.24676768, 6.58060606, 6.91444444, 7.24828283, 7.58212121, 7.91595960, 8.24979798,
           8.58363636, 8.91747475, 9.25131313, 9.58515152, 9.91898990, 10.25282828, 10.58666667, 10.92050505, 11.25434343,
           11.58818182, 11.92202020, 12.25585859, 12.58969697, 12.92353535, 13.25737374, 13.59121212, 13.92505051,
           14.25888889, 14.59272727, 14.92656566, 15.26040404, 15.59424242, 15.92808081, 16.26191919, 16.59575758,
           16.92959596, 17.26343434, 17.59727273, 17.93111111, 18.26494949, 18.59878788, 18.93262626, 19.26646465,
           19.60030303, 19.93414141, 20.26797980, 20.60181818, 20.93565657, 21.26949495, 21.60333333, 21.93717172,
           22.27101010, 22.60484848, 22.93868687, 23.27252525, 23.60636364, 23.94020202, 24.27404040, 24.60787879,
           24.94171717, 25.27555556, 25.60939394, 25.94323232, 26.27707071, 26.61090909, 26.94474747, 27.27858586,
           27.61242424, 27.94626263, 28.28010101, 28.61393939, 28.94777778, 29.28161616, 29.61545455, 29.94929293,
           30.28313131, 30.61696970, 30.95080808, 31.28464646, 31.61848485, 31.95232323, 32.28616162, 32.62000000),
  fit = c(-0.910193553, -0.892328263, -0.874463854, -0.856601239, -0.838741484, -0.820885983, -0.803036232,
          -0.785193657, -0.767359440, -0.749534192, -0.731717896, -0.713909826, -0.696108628, -0.678312105,
          -0.660517350, -0.642720801, -0.624917169, -0.607099562, -0.589259913, -0.571389493, -0.553479352,
          -0.535520781, -0.517505583, -0.499426445, -0.481277779, -0.463056732, -0.444765518, -0.426413341,
          -0.408016065, -0.389594256, -0.371172261, -0.352778851, -0.334445852, -0.316206540, -0.298093711,
          -0.280137398, -0.262363286, -0.244791649, -0.227435436, -0.210299867, -0.193383223, -0.176678103,
          -0.160173109, -0.143854523, -0.127708968, -0.111725789, -0.095900227, -0.080236846, -0.064750951,
          -0.049470085, -0.034434514, -0.019695835, -0.005313855, 0.008646802, 0.022120309, 0.035043794, 0.047361091,
          0.059026494, 0.070007592, 0.080286832, 0.089861857, 0.098744217, 0.106957399, 0.114534402, 0.121514725,
          0.127940795, 0.133854855, 0.139296239, 0.144299609, 0.148893445, 0.153099391, 0.156932978, 0.160405196,
          0.163524177, 0.166297123, 0.168732430, 0.170841733, 0.172641098, 0.174151237, 0.175397434, 0.176409009,
          0.177218616, 0.177861187, 0.178372239, 0.178786015, 0.179133417, 0.179440298, 0.179727109, 0.180008131,
          0.180291815, 0.180583397, 0.180885638, 0.181198748, 0.181521954, 0.181853745, 0.182192430, 0.182536264,
          0.182883491, 0.183232606, 0.183582506),
  se = c(0.17211954, 0.16311094, 0.15428671, 0.14568448, 0.13734684, 0.12932314, 0.12166542, 0.11442295, 0.10763669,
         0.10133602, 0.09553482, 0.09022920, 0.08539656, 0.08099750, 0.07697916, 0.07328074, 0.06984581, 0.06663059,
         0.06360814, 0.06076819, 0.05811340, 0.05565202, 0.05339007, 0.05132463, 0.04944060, 0.04771044, 0.04610018,
         0.04457828, 0.04312182, 0.04171777, 0.04036451, 0.03907493, 0.03787100, 0.03677386, 0.03579382, 0.03492732,
         0.03415490, 0.03344412, 0.03276060, 0.03207847, 0.03138644, 0.03068832, 0.02999939, 0.02934002, 0.02872413,
         0.02815031, 0.02760271, 0.02706039, 0.02650702, 0.02594124, 0.02538174, 0.02486241, 0.02441859, 0.02407221,
         0.02382322, 0.02365038, 0.02352035, 0.02340317, 0.02328596, 0.02317818, 0.02310674, 0.02310238, 0.02318496,
         0.02335481, 0.02359390, 0.02387607, 0.02418379, 0.02451898, 0.02490329, 0.02537107, 0.02595516, 0.02667037,
         0.02750464, 0.02842421, 0.02938966, 0.03037387, 0.03137654, 0.03242785, 0.03358062, 0.03489720, 0.03643498,
         0.03823661, 0.04032799, 0.04272410, 0.04543907, 0.04849785, 0.05194228, 0.05582673, 0.06020903, 0.06513773,
         0.07063984, 0.07671707, 0.08334793, 0.09049070, 0.09808965, 0.10607860, 0.11438690, 0.12294885, 0.13170865,
         0.14062195)
)

# Plotly version of the smooth effect
temp_smooth <- plot_ly(df_temp, 
                       x = ~temp,
                       y = ~fit, 
                       type = 'scatter', 
                       mode = 'lines',
                       line = list(color = "#0072B2"), name = "Smooth effect",
                       hovertemplate = paste(
                         "Temp.: %{x:.1f}°C<br>",
                         "Effect: %{y:.2f}<extra></extra>"
                       )) |>
  add_ribbons(ymin = ~fit - 2*se, ymax = ~fit + 2*se,
              line = list(color = "transparent"),
              fillcolor = "rgba(0,114,178,0.3)",
              name = "95% CI") |>
  layout(
    title = list(
      text = "Smooth Effect of Temperature",
      x = 0.5,
      font = list(size = 18, family = "Arial")
    ),
    xaxis = list(
      title = "Temperature (°C)",
      titlefont = list(size = 14),
      tickfont = list(size = 12),
      showgrid = TRUE, 
      gridcolor = "lightgray"
    ),
    yaxis = list(
      title = "Smooth Effect",
      titlefont = list(size = 14),
      tickfont = list(size = 12),
      showgrid = TRUE, 
      gridcolor = "lightgray"
    ),
    legend = list(x = 0.02, y = 0.98),
    margin = list(l = 80, r = 40, t = 80, b = 60),
    paper_bgcolor = "#ffffff",
    plot_bgcolor = "#ffffff"
  )

# Random Forest
rf_train <- train_df |> 
  mutate(hour = as.factor(hour),
         weather_main = as.factor(weather_main)) |> 
  drop_na()
rf_test <- test_df |>
  mutate(hour = as.factor(hour),
         weather_main = as.factor(weather_main)) |>
  drop_na()

rf_formula <- total_trips ~ temp + pressure + humidity + wind_speed +
  cloudiness + hour + weather_main

default_rf <- randomForest(formula = rf_formula, data = rf_train, importance = TRUE)

rf_train$predicted <- predict(default_rf, newdata = rf_train)
rf_test$predicted  <- predict(default_rf, newdata = rf_test)

metrics_train_rf_default <- metrics(rf_train, truth = total_trips, estimate = predicted)
metrics_test_rf_default  <- metrics(rf_test,  truth = total_trips, estimate = predicted)

imp_raw <- importance(default_rf)
importance_df <- data.frame(
  Variable = rownames(imp_raw),
  Importance = imp_raw[, "%IncMSE"]
)

importance_df <- importance_df |>
  arrange(Importance) |>
  mutate(
    Label = paste0(round(Importance, 1), "%"),
    Variable = factor(Variable, levels = Variable)
  )

# Plot with enhancements
default_rf_var_importance <- plot_ly(
  data = importance_df,
  x = ~Importance,
  y = ~Variable,
  type = "bar",
  orientation = "h",
  text = ~Label,
  textposition = "inside",
  insidetextanchor = "start",  # move text slightly inside
  textfont = list(color = "black", size = 12),
  marker = list(
    color = ~Importance,
    colorscale = list(
      c(0, "#FFFFB2"), c(0.25, "#FECC5C"),
      c(0.5, "#FD8D3C"), c(0.75, "#F03B20"), c(1, "#BD0026")
    ),
    line = list(color = "black", width = 0.5)
  ),
  hovertemplate = paste(
    "<b>%{y}</b><br>",
    "Importance: %{x:.1f}%<extra></extra>"
  ),
  height = 500
) |>
  layout(
    title = list(
      text = paste0("Feature Importance for Random Forest (Default)<br>"),
      x = 0.5,
      font = list(size = 18)
    ),
    xaxis = list(title = "Importance (%IncMSE)", titlefont = list(size = 14)),
    yaxis = list(
      title = "",
      tickfont = list(size = 14),
      tickpadding = 20,  # Add space between y-axis and y labels
      automargin = TRUE  # Ensures enough margin is added if needed
    ),
    margin = list(l = 180, r = 40, t = 100, b = 50),
    showlegend = FALSE,
    paper_bgcolor = "#ffffff",
    plot_bgcolor = "#ffffff"
  )

final_rf <- randomForest(
  rf_formula,
  data = rf_train,
  mtry = 4,
  ntree = 500,
  nodesize = 10,
  importance=TRUE
)

rf_train$predicted <- predict(final_rf, newdata = rf_train)
rf_test$predicted  <- predict(final_rf, newdata = rf_test)

metrics_train_rf <- yardstick::metrics(rf_train, truth = total_trips, estimate = predicted)
metrics_test_rf  <- yardstick::metrics(rf_test,  truth = total_trips, estimate = predicted)


imp_raw <- importance(final_rf)
importance_df <- data.frame(
  Variable = rownames(imp_raw),
  Importance = imp_raw[, "%IncMSE"]
)

importance_df <- importance_df |>
  arrange(Importance) |>
  mutate(
    Label = paste0(round(Importance, 1), "%"),
    Variable = factor(Variable, levels = Variable)
  )

# Plot with enhancements
final_rf_var_importance <- plot_ly(
  data = importance_df,
  x = ~Importance,
  y = ~Variable,
  type = "bar",
  orientation = "h",
  text = ~Label,
  textposition = "inside",
  insidetextanchor = "start",  # move text slightly inside
  textfont = list(color = "black", size = 12),
  marker = list(
    color = ~Importance,
    colorscale = list(
      c(0, "#FFFFB2"), c(0.25, "#FECC5C"),
      c(0.5, "#FD8D3C"), c(0.75, "#F03B20"), c(1, "#BD0026")
    ),
    line = list(color = "black", width = 0.5)
  ),
  hovertemplate = paste(
    "<b>%{y}</b><br>",
    "Importance: %{x:.1f}%<extra></extra>"
  ),
  height = 500
) |>
  layout(
    title = list(
      text = paste0("Feature Importance for Random Forest (Cross-Validated)<br>",
                    "<sup>mtry = ", 4,
                    ", ntree = ", 500,
                    ", nodesize = ", 10, "</sup>"),
      x = 0.5,
      font = list(size = 18)
    ),
    xaxis = list(title = "Importance (%IncMSE)", titlefont = list(size = 14)),
    yaxis = list(
      title = "",
      tickfont = list(size = 14),
      tickpadding = 20,  # Add space between y-axis and y labels
      automargin = TRUE  # Ensures enough margin is added if needed
    ),
    margin = list(l = 180, r = 40, t = 100, b = 50),
    showlegend = FALSE,
    paper_bgcolor = "#ffffff",
    plot_bgcolor = "#ffffff"
  )

# XGBoost Model
xgb_train <- train_df |>
  mutate(hour = as.numeric(hour),
         weather_main = as.numeric(as.factor(weather_main))) |>
  drop_na()

xgb_test <- test_df |>
  mutate(hour = as.numeric(hour),
         weather_main = as.numeric(as.factor(weather_main))) |>
  drop_na()

train_matrix <- model.matrix(total_trips ~ . - weather_desc - precipitation - feels_like - date - 1, data = xgb_train)
train_label  <- xgb_train$total_trips

test_matrix  <- model.matrix(total_trips ~ . - weather_desc - precipitation - feels_like - date - 1, data = xgb_test)
test_label  <- xgb_test$total_trips

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)

default_xgb <- xgboost(
  data = dtrain,
  objective = "reg:squarederror",
  nrounds = 100,
  verbose = 0
)

pred_train <- predict(default_xgb, dtrain)
pred_test  <- predict(default_xgb, dtest)

metrics_train_xgb_default <- yardstick::metrics(data.frame(truth = train_label, estimate = pred_train),
                                                truth = truth, estimate = estimate)
metrics_test_xgb_default <- yardstick::metrics(data.frame(truth = test_label, estimate = pred_test),
                                               truth = truth, estimate = estimate)

xgb_imp <- xgb.importance(model = default_xgb) %>%
  arrange(Gain) %>%
  mutate(
    Feature = factor(Feature, levels = Feature),
    Label = paste0(round(Gain * 100, 1), "%")
  )

# Plot
default_xgb_var_importance <- plot_ly(
  data = xgb_imp,
  x = ~Gain,
  y = ~reorder(Feature, Gain),
  type = "bar",
  orientation = "h",
  text = ~Label,
  textposition = "outside",
  marker = list(
    color = ~Gain,
    colorscale = list(
      c(0, "#e0f3db"),
      c(0.33, "#a8ddb5"),
      c(0.66, "#43a2ca"),
      c(1, "#006d2c")
    ),
    line = list(color = "black", width = 0.5)
  ),
  hovertemplate = paste(
    "<b>%{y}</b><br>",
    "Gain: %{x:.4f}<extra></extra>"
  )
) %>%
  layout(
    title = list(
      text = "Feature Importance for XGBoost (Default)",
      x = 0.5,
      y = 0.9
    ),
    margin = list(t = 100),  # make room for annotation
    xaxis = list(title = "Importance (Gain)"),
    yaxis = list(title = ""),
    showlegend = FALSE
  )

final_xgb <- xgboost(
  params = list(
    eta = 0.05,
    max_depth = 6,
    min_child_weight = 3,
    subsample = 0.8,
    colsample_bytree = 1,
    objective = "reg:squarederror"
  ),
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

xgb_imp <- xgb.importance(model = final_xgb) %>%
  arrange(Gain) %>%
  mutate(
    Feature = factor(Feature, levels = Feature),
    Label = paste0(round(Gain * 100, 1), "%")
  )

# Plot
final_xgb_var_importance <- plot_ly(
  data = xgb_imp,
  x = ~Gain,
  y = ~reorder(Feature, Gain),
  type = "bar",
  orientation = "h",
  text = ~Label,
  textposition = "outside",
  marker = list(
    color = ~Gain,
    colorscale = list(
      c(0, "#e0f3db"),
      c(0.33, "#a8ddb5"),
      c(0.66, "#43a2ca"),
      c(1, "#006d2c")
    ),
    line = list(color = "black", width = 0.5)
  ),
  hovertemplate = paste(
    "<b>%{y}</b><br>",
    "Gain: %{x:.4f}<extra></extra>"
  )
) %>%
  layout(
    title = list(
      text = "Feature Importance for XGBoost (Cross-Validated)",
      x = 0.5,
      y = 0.9
    ),
    annotations = list(
      list(
        text = "eta = 0.05, max_depth = 6, min_child_weight = 3, subsample = 0.8, colsample_bytree = 1",
        x = 0.5,
        y = 1.08,
        xref = "paper",
        yref = "paper",
        showarrow = FALSE,
        font = list(size = 12),
        align = "center"
      )
    ),
    margin = list(t = 100),  # make room for annotation
    xaxis = list(title = "Importance (Gain)"),
    yaxis = list(title = ""),
    showlegend = FALSE
  )

final_xgb_var_importance2 <- plot_ly(
  data = xgb_imp,
  x = ~Gain,
  y = ~reorder(Feature, Gain),
  type = "bar",
  orientation = "h",
  text = ~Label,
  textposition = "outside",
  marker = list(
    color = ~Gain,
    colorscale = list(
      c(0, "#e0f3db"),
      c(0.33, "#a8ddb5"),
      c(0.66, "#43a2ca"),
      c(1, "#006d2c")
    ),
    line = list(color = "black", width = 0.5)
  ),
  hovertemplate = paste(
    "<b>%{y}</b><br>",
    "Gain: %{x:.4f}<extra></extra>"
  )
) %>%
  layout(
    title = list(
      text = "Feature Importance for XGBoost (Cross-Validated)",
      x = 0.5,
      y = 0.9
    ),
    margin = list(t = 100),  # make room for annotation
    xaxis = list(title = "Importance (Gain)"),
    yaxis = list(title = ""),
    showlegend = FALSE
  )


# Pred vs Actual

pred_vs_actual <- tibble(
  actual = rf_test$total_trips,
  predicted = predict(final_rf, newdata = rf_test)
) %>%
  mutate(abs_error = abs(actual - predicted))

# Compute R²
rsq <- cor(pred_vs_actual$actual, pred_vs_actual$predicted)^2
pred_vs_actual <- pred_vs_actual %>%
  mutate(residual = abs(predicted - actual))

axis_min <- min(pred_vs_actual$actual, pred_vs_actual$predicted)
axis_max <- max(pred_vs_actual$actual, pred_vs_actual$predicted)

final_pred <- plot_ly() %>%
  add_trace(
    data = pred_vs_actual,
    x = ~actual,
    y = ~predicted,
    type = "scatter",
    mode = "markers",
    marker = list(
      color = ~residual,
      colorscale = "YlOrRd",
      colorbar = list(title = "Absolute Error"),
      opacity = 0.6,
      size = 6
    ),
    text = ~paste("Actual:", actual,
                  "<br>Predicted:", round(predicted, 1),
                  "<br>Absolute Error:", round(residual, 1)),
    hoverinfo = "text",
    name = "Predictions"
  ) %>%
  add_trace(
    x = c(axis_min, axis_max),
    y = c(axis_min, axis_max),
    type = "scatter",
    mode = "lines",
    line = list(dash = "dash", color = "gray"),
    showlegend = FALSE,
    hoverinfo = "none"
  ) %>%
  layout(
    title = list(text = "Predicted vs. Actual Bike Trip Counts<br><sup>Colored by Absolute Error (Random Forest)</sup>", x = 0.5),
    xaxis = list(title = "Actual Bike Trips"),
    yaxis = list(title = "Predicted Bike Trips"),
    hovermode = "closest"
  )

# Kable Comparison
model_perf <- data.frame(
  Model = c("Linear Regression (LM)", 
            "GLM (Negative Binomial)", 
            "GAM (Negative Binomial)", 
            "Random Forest (Default)", 
            "Random Forest (CV)", 
            "XGBoost (Default)", 
            "XGBoost (CV)"),
  
  R2_Train = c(0.826, 0.817, 0.829, 0.965, 0.952, 0.978, 0.905),
  RMSE_Train = c(360.692, 368.451, 355.891, 174.114, 193.634, 129.561, 268.267),
  MAE_Train = c(251.108, 254.049, 246.153, 122.431, 132.163, 85.818, 184.694),
  
  R2_Test = c(0.830, 0.819, 0.833, 0.845, 0.846, 0.830, 0.845),
  RMSE_Test = c(347.020, 359.205, 343.998, 336.344, 330.028, 347.255, 331.139),
  MAE_Test = c(245.548, 252.532, 246.577, 250.219, 236.966, 247.811, 238.871)
)

model_perf %>%
  kbl(
    col.names = c("Model", "$R^2$", "RMSE", "MAE", "$R^2$", "RMSE", "MAE"),
    align = c("l", "c", "c", "c", "c", "c", "c"),
    booktabs = TRUE,
    escape = FALSE
  ) %>%
  add_header_above(c(" " = 1, "Train Set" = 3, "Test Set" = 3), bold = TRUE) %>%
  row_spec(5, bold = TRUE) %>%
  kable_styling(full_width = FALSE, position = "center")