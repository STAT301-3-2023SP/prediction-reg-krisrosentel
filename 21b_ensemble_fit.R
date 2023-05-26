# load libraries
library(pacman)
p_load(tidymodels, tidyverse, stacks, MASS, poissonreg, 
       mgcv, kknn, ranger, xgboost, earth, nnet, kernlab, baguette)

# deal with package conflicts
tidymodels_prefer()

# Due to memory issues, fit members doesn't work so we need to fit each member 
# individually and then combine preds using the weights

# load weights and data
load("results/ensemble_wts.rda") # 290 candidate members, penalty = .9
reg_train <- read_csv("data/train.csv") 
reg_test <- read_csv("data/test.csv")

# Load cv
load("results/elastic_nb_int_cv.rda")

# fit and predict
elastic_nb_int_fit_1_06_wf <- elastic_nb_int_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_nb_int_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model06"))

elastic_nb_int_fit_1_06_res <- fit(elastic_nb_int_fit_1_06_wf, reg_train)

pred1 <- predict(elastic_nb_int_fit_1_06_res, new_data = reg_test) %>% 
  rename(elastic_nb_int_fit_1_06 = .pred) %>% 
  mutate(elastic_nb_int_fit_1_06wt = stack_coef %>% 
           filter(member == "elastic_nb_int_fit_1_06") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(elastic_nb_int_fit, elastic_nb_int_tictoc, elastic_nb_int_fit_1_06_wf, 
   elastic_nb_int_fit_1_06_res)

# Load cv
load("results/elastic_nb_pca_cv.rda")

# fit and predict
elastic_nb_pca_fit_1_06_wf <- elastic_nb_pca_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_nb_pca_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model06"))

elastic_nb_pca_fit_1_06_res <- fit(elastic_nb_pca_fit_1_06_wf, reg_train)

pred2 <- predict(elastic_nb_pca_fit_1_06_res, new_data = reg_test) %>% 
  rename(elastic_nb_pca_fit_1_06 = .pred) %>% 
  mutate(elastic_nb_pca_fit_1_06wt = stack_coef %>% 
           filter(member == "elastic_nb_pca_fit_1_06") %>% 
           select(weight) %>% 
           unlist())

# fit and predict
elastic_nb_pca_fit_1_07_wf <- elastic_nb_pca_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_nb_pca_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model07"))

elastic_nb_pca_fit_1_07_res <- fit(elastic_nb_pca_fit_1_07_wf, reg_train)

pred3 <- predict(elastic_nb_pca_fit_1_07_res, new_data = reg_test) %>% 
  rename(elastic_nb_pca_fit_1_07 = .pred) %>% 
  mutate(elastic_nb_pca_fit_1_07wt = stack_coef %>% 
           filter(member == "elastic_nb_pca_fit_1_07") %>% 
           select(weight) %>% 
           unlist())

# fit and predict
elastic_nb_pca_fit_1_12_wf <- elastic_nb_pca_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_nb_pca_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model12"))

elastic_nb_pca_fit_1_12_res <- fit(elastic_nb_pca_fit_1_12_wf, reg_train)

pred4 <- predict(elastic_nb_pca_fit_1_12_res, new_data = reg_test) %>% 
  rename(elastic_nb_pca_fit_1_12 = .pred) %>% 
  mutate(elastic_nb_pca_fit_1_12wt = stack_coef %>% 
           filter(member == "elastic_nb_pca_fit_1_12") %>% 
           select(weight) %>% 
           unlist())

# fit and predict
elastic_nb_pca_fit_1_18_wf <- elastic_nb_pca_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_nb_pca_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model18"))

elastic_nb_pca_fit_1_18_res <- fit(elastic_nb_pca_fit_1_18_wf, reg_train)

pred5 <- predict(elastic_nb_pca_fit_1_18_res, new_data = reg_test) %>% 
  rename(elastic_nb_pca_fit_1_18 = .pred) %>% 
  mutate(elastic_nb_pca_fit_1_18wt = stack_coef %>% 
           filter(member == "elastic_nb_pca_fit_1_18") %>% 
           select(weight) %>% 
           unlist())

# fit and predict
elastic_nb_pca_fit_1_24_wf <- elastic_nb_pca_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_nb_pca_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model24"))

elastic_nb_pca_fit_1_24_res <- fit(elastic_nb_pca_fit_1_24_wf, reg_train)

pred6 <- predict(elastic_nb_pca_fit_1_24_res, new_data = reg_test) %>% 
  rename(elastic_nb_pca_fit_1_24 = .pred) %>% 
  mutate(elastic_nb_pca_fit_1_24wt = stack_coef %>% 
           filter(member == "elastic_nb_pca_fit_1_24") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(elastic_nb_pca_fit, elastic_nb_pca_tictoc, elastic_nb_pca_fit_1_06_wf, 
   elastic_nb_pca_fit_1_06_res, elastic_nb_pca_fit_1_07_wf, 
   elastic_nb_pca_fit_1_07_res, elastic_nb_pca_fit_1_12_wf, 
   elastic_nb_pca_fit_1_12_res, elastic_nb_pca_fit_1_18_wf,
   elastic_nb_pca_fit_1_18_res, elastic_nb_pca_fit_1_24_f,
   elastic_nb_pca_fit_1_24_res)

# Load cv
load("results/svm_rad_cv.rda")

# fit and predict
svm_rad_fit_1_15_wf <- svm_rad_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(svm_rad_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model15"))

svm_rad_fit_1_15_res <- fit(svm_rad_fit_1_15_wf, reg_train)

pred7 <- predict(svm_rad_fit_1_15_res, new_data = reg_test) %>% 
  rename(svm_rad_fit_1_15 = .pred) %>% 
  mutate(svm_rad_fit_1_15wt = stack_coef %>% 
           filter(member == "svm_rad_fit_1_15") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(svm_rad_fit, svm_rad_tictoc, svm_rad_fit_1_15_wf, svm_rad_fit_1_15_res)

# Load cv
load("results/mlp_cv.rda")

# fit and predict
mlp_fit_1_06_wf <- mlp_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(mlp_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model06"))

mlp_fit_1_06_res <- fit(mlp_fit_1_06_wf, reg_train)

pred8 <- predict(mlp_fit_1_06_res, new_data = reg_test) %>% 
  rename(mlp_fit_1_06 = .pred) %>% 
  mutate(mlp_fit_1_06wt = stack_coef %>% 
           filter(member == "mlp_fit_1_06") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(mlp_fit, mlp_tictoc, mlp_fit_1_06_wf, mlp_fit_1_06_res)

# Load cv
load("results/mars_cv.rda")

# fit and predict
mars_fit_1_52_wf <- mars_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(mars_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model52"))

mars_fit_1_52_res <- fit(mars_fit_1_52_wf, reg_train)

pred9 <- predict(mars_fit_1_52_res, new_data = reg_test) %>% 
  rename(mars_fit_1_52 = .pred) %>% 
  mutate(mars_fit_1_52wt = stack_coef %>% 
           filter(member == "mars_fit_1_52") %>% 
           select(weight) %>% 
           unlist())

# fit and predict
mars_fit_1_59_wf <- mars_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(mars_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model59"))

mars_fit_1_59_res <- fit(mars_fit_1_59_wf, reg_train)

pred10 <- predict(mars_fit_1_59_res, new_data = reg_test) %>% 
  rename(mars_fit_1_59 = .pred) %>% 
  mutate(mars_fit_1_59wt = stack_coef %>% 
           filter(member == "mars_fit_1_59") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(mars_fit, mars_tictoc, mars_fit_1_52_wf, mars_fit_1_52_res,
   mars_fit_1_59_wf, mars_fit_1_59_res)

# write function to restore bounds of response variable 
restore_bounds <- function(x){ 
  x <- x %>% mutate(y = case_when(y < 1 ~ 1,
                                  y > 100 ~ 100,
                                  .default = y))
}

# Combine into dataframe
ensemble_preds <- reg_test %>% 
  select(id) %>% 
  cbind(pred1) %>% 
  cbind(pred2) %>% 
  cbind(pred3) %>% 
  cbind(pred4) %>% 
  cbind(pred5) %>% 
  cbind(pred6) %>% 
  cbind(pred7) %>% 
  cbind(pred8) %>% 
  cbind(pred9) %>% 
  cbind(pred10) %>% 
  mutate(y = elastic_nb_int_fit_1_06 * elastic_nb_int_fit_1_06wt +
           elastic_nb_pca_fit_1_06 * elastic_nb_pca_fit_1_06wt +
           elastic_nb_pca_fit_1_07 * elastic_nb_pca_fit_1_07wt +
           elastic_nb_pca_fit_1_12 * elastic_nb_pca_fit_1_12wt +
           elastic_nb_pca_fit_1_18 * elastic_nb_pca_fit_1_18wt +
           elastic_nb_pca_fit_1_24 * elastic_nb_pca_fit_1_24wt +
           svm_rad_fit_1_15 * svm_rad_fit_1_15wt +
           mlp_fit_1_06 * mlp_fit_1_06wt +
           mars_fit_1_52 * mars_fit_1_52wt +
           mars_fit_1_59 * mars_fit_1_59wt) %>% 
  restore_bounds()

# Save
save(ensemble_preds,
     file = "results/ensemble_preds.rda")