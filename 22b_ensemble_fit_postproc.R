# load libraries
library(pacman)
p_load(tidymodels, tidyverse, stacks, MASS, poissonreg, 
       mgcv, kknn, ranger, xgboost, earth, nnet, kernlab, baguette)

# deal with package conflicts
tidymodels_prefer()
set.seed(900) # set seed 

# Due to memory issues, fit members doesn't work so we need to fit each member 
# individually and then combine preds using the weights

# load weights and data
load("results/ensemble2_wts.rda") # 290 candidate members, penalty = .2
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
  mutate(elastic_nb_int_fit_1_06wt = stack_coef2 %>% 
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
  mutate(elastic_nb_pca_fit_1_06wt = stack_coef2 %>% 
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
  mutate(elastic_nb_pca_fit_1_07wt = stack_coef2 %>% 
           filter(member == "elastic_nb_pca_fit_1_07") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(elastic_nb_pca_fit, elastic_nb_pca_tictoc, elastic_nb_pca_fit_1_06_wf, 
   elastic_nb_pca_fit_1_06_res, elastic_nb_pca_fit_1_07_wf, 
   elastic_nb_pca_fit_1_07_res)

# Load cv
load("results/svm_rad_cv.rda")

# fit and predict
svm_rad_fit_1_15_wf <- svm_rad_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(svm_rad_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model15"))

svm_rad_fit_1_15_res <- fit(svm_rad_fit_1_15_wf, reg_train)

pred4 <- predict(svm_rad_fit_1_15_res, new_data = reg_test) %>% 
  rename(svm_rad_fit_1_15 = .pred) %>% 
  mutate(svm_rad_fit_1_15wt = stack_coef2 %>% 
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

pred5 <- predict(mlp_fit_1_06_res, new_data = reg_test) %>% 
  rename(mlp_fit_1_06 = .pred) %>% 
  mutate(mlp_fit_1_06wt = stack_coef2 %>% 
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

pred6 <- predict(mars_fit_1_52_res, new_data = reg_test) %>% 
  rename(mars_fit_1_52 = .pred) %>% 
  mutate(mars_fit_1_52wt = stack_coef2 %>% 
           filter(member == "mars_fit_1_52") %>% 
           select(weight) %>% 
           unlist())

# fit and predict
mars_fit_1_59_wf <- mars_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(mars_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model59"))

mars_fit_1_59_res <- fit(mars_fit_1_59_wf, reg_train)

pred7 <- predict(mars_fit_1_59_res, new_data = reg_test) %>% 
  rename(mars_fit_1_59 = .pred) %>% 
  mutate(mars_fit_1_59wt = stack_coef2 %>% 
           filter(member == "mars_fit_1_59") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(mars_fit, mars_tictoc, mars_fit_1_52_wf, mars_fit_1_52_res,
   mars_fit_1_59_wf, mars_fit_1_59_res)

# write function to restore bounds of response variable 
restore_bounds <- function(x){ 
  x <- case_when(x < 1 ~ 1,
                 x > 100 ~ 100,
                 .default = x)
  x
}

# Combine into dataframe
ensemble_preds2 <- reg_test %>% 
  select(id) %>% 
  cbind(pred1) %>% 
  cbind(pred2) %>% 
  cbind(pred3) %>% 
  cbind(pred4) %>% 
  cbind(pred5) %>% 
  cbind(pred6) %>% 
  cbind(pred7) %>% 
  mutate_at(c("elastic_nb_int_fit_1_06", "elastic_nb_pca_fit_1_06",
              "elastic_nb_pca_fit_1_07", "svm_rad_fit_1_15", "mlp_fit_1_06",
              "mars_fit_1_52", "mars_fit_1_59"), restore_bounds) %>% 
  mutate(y = elastic_nb_int_fit_1_06 * elastic_nb_int_fit_1_06wt +
           elastic_nb_pca_fit_1_06 * elastic_nb_pca_fit_1_06wt +
           elastic_nb_pca_fit_1_07 * elastic_nb_pca_fit_1_07wt +
           svm_rad_fit_1_15 * svm_rad_fit_1_15wt +
           mlp_fit_1_06 * mlp_fit_1_06wt +
           mars_fit_1_52 * mars_fit_1_52wt +
           mars_fit_1_59 * mars_fit_1_59wt) 

# Save
save(ensemble_preds2,
     file = "results/ensemble_preds2.rda")