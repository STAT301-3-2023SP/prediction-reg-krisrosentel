# load libraries
library(pacman)
p_load(tidymodels, tidyverse, stacks, MASS, poissonreg, 
       mgcv, kknn, ranger, xgboost, earth, nnet, kernlab)

# deal with package conflicts
tidymodels_prefer()

# Due to memory issues, fit members doesn't work so we need to fit each member 
# individually and then combine preds using the weights

# load weights and data
load("results/ensemble_wts.rda")
reg_train <- read_csv("data/train.csv") 
reg_test <- read_csv("data/test.csv")

# Load cv
load("results/elastic_psn_cv.rda")

# fit and predict
elastic_psn_fit_1_05_wf <- elastic_psn_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_psn_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model05"))

elastic_psn_fit_1_05_res <- fit(elastic_psn_fit_1_05_wf, reg_train)

pred1 <- predict(elastic_psn_fit_1_05_res, new_data = reg_test) %>% 
  rename(elastic_psn_fit_1_05 = .pred) %>% 
  mutate(elastic_psn_fit_1_05wt = stack_coef %>% 
           filter(member == "elastic_psn_fit_1_05") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(elastic_psn_fit, elastic_psn_tictoc, elastic_psn_fit_1_05_wf, 
   elastic_psn_fit_1_05_res)

# Load cv
load("results/elastic_lm_int_cv.rda")

# fit and predict
elastic_lm_int_fit_1_06_wf <- elastic_lm_int_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_lm_int_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model06"))

elastic_lm_int_fit_1_06_res <- fit(elastic_lm_int_fit_1_06_wf, reg_train)

pred2 <- predict(elastic_lm_int_fit_1_06_res, new_data = reg_test) %>% 
  rename(elastic_lm_int_fit_1_06 = .pred) %>% 
  mutate(elastic_lm_int_fit_1_06wt = stack_coef %>% 
           filter(member == "elastic_lm_int_fit_1_06") %>% 
           select(weight) %>% 
           unlist())

# fit and predict
elastic_lm_int_fit_1_11_wf <- elastic_lm_int_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_lm_int_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model11"))

elastic_lm_int_fit_1_11_res <- fit(elastic_lm_int_fit_1_11_wf, reg_train)

pred3 <- predict(elastic_lm_int_fit_1_11_res, new_data = reg_test) %>% 
  rename(elastic_lm_int_fit_1_11 = .pred) %>% 
  mutate(elastic_lm_int_fit_1_11wt = stack_coef %>% 
           filter(member == "elastic_lm_int_fit_1_11") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(elastic_lm_int_fit, elastic_lm_int_tictoc, elastic_lm_int_fit_1_06_wf, 
   elastic_lm_int_fit_1_06_res, elastic_lm_int_fit_1_11_wf, 
   elastic_lm_int_fit_1_11_res)

# Load cv
load("results/elastic_psn_int_cv.rda")

# fit and predict
elastic_psn_int_fit_1_05_wf <- elastic_psn_int_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_psn_int_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model05"))

elastic_psn_int_fit_1_05_res <- fit(elastic_psn_int_fit_1_05_wf, reg_train)

pred4 <- predict(elastic_psn_int_fit_1_05_res, new_data = reg_test) %>% 
  rename(elastic_psn_int_fit_1_05 = .pred) %>% 
  mutate(elastic_psn_int_fit_1_05wt = stack_coef %>% 
           filter(member == "elastic_psn_int_fit_1_05") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(elastic_psn_int_fit, elastic_psn_int_tictoc, elastic_psn_int_fit_1_05_wf, 
   elastic_psn_int_fit_1_05_res)

# Load cv
load("results/elastic_nb_int_cv.rda")

# fit and predict
elastic_nb_int_fit_1_05_wf <- elastic_nb_int_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_nb_int_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model05"))

elastic_nb_int_fit_1_05_res <- fit(elastic_nb_int_fit_1_05_wf, reg_train)

pred5 <- predict(elastic_nb_int_fit_1_05_res, new_data = reg_test) %>% 
  rename(elastic_nb_int_fit_1_05 = .pred) %>% 
  mutate(elastic_nb_int_fit_1_05wt = stack_coef %>% 
           filter(member == "elastic_nb_int_fit_1_05") %>% 
           select(weight) %>% 
           unlist())

# fit and predict
elastic_nb_int_fit_1_06_wf <- elastic_nb_int_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(elastic_nb_int_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model06"))

elastic_nb_int_fit_1_06_res <- fit(elastic_nb_int_fit_1_06_wf, reg_train)

pred6 <- predict(elastic_nb_int_fit_1_06_res, new_data = reg_test) %>% 
  rename(elastic_nb_int_fit_1_06 = .pred) %>% 
  mutate(elastic_nb_int_fit_1_06wt = stack_coef %>% 
           filter(member == "elastic_nb_int_fit_1_06") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(elastic_nb_int_fit, elastic_nb_int_tictoc, elastic_nb_int_fit_1_05_wf, 
   elastic_nb_int_fit_1_05_res, elastic_nb_int_fit_1_06_wf, 
   elastic_nb_int_fit_1_06_res)

# Load cv
load("results/gam_nb_cv.rda")

# fit and predict
gam_nb_fit_1_1_wf <- gam_nb_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(gam_nb_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model1"))

gam_nb_fit_1_1_res <- fit(gam_nb_fit_1_1_wf, reg_train)

pred7 <- predict(gam_nb_fit_1_1_res, new_data = reg_test) %>% 
  rename(gam_nb_fit_1_1 = .pred) %>% 
  mutate(gam_nb_fit_1_1wt = stack_coef %>% 
           filter(member == "gam_nb_fit_1_1") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(gam_nb_fit, gam_nb_tictoc, gam_nb_fit_1_1_wf, gam_nb_fit_1_1_res)

# Load cv
load("results/svm_rad_cv.rda")

# fit and predict
svm_rad_fit_1_6_wf <- svm_rad_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(svm_rad_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model6"))

svm_rad_fit_1_6_res <- fit(svm_rad_fit_1_6_wf, reg_train)

pred8 <- predict(svm_rad_fit_1_6_res, new_data = reg_test) %>% 
  rename(svm_rad_fit_1_6 = .pred) %>% 
  mutate(svm_rad_fit_1_6wt = stack_coef %>% 
           filter(member == "svm_rad_fit_1_6") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(svm_rad_fit, svm_rad_tictoc, svm_rad_fit_1_6_wf, svm_rad_fit_1_6_res)

# Load cv
load("results/bt_cv.rda")

# fit and predict
bt_fit_1_35_wf <- bt_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(bt_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model35"))

bt_fit_1_35_res <- fit(bt_fit_1_35_wf, reg_train)

pred9 <- predict(bt_fit_1_35_res, new_data = reg_test) %>% 
  rename(bt_fit_1_35 = .pred) %>% 
  mutate(bt_fit_1_35wt = stack_coef %>% 
           filter(member == "bt_fit_1_35") %>% 
           select(weight) %>% 
           unlist())

# fit and predict
bt_fit_1_39_wf <- bt_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(bt_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model39"))

bt_fit_1_39_res <- fit(bt_fit_1_39_wf, reg_train)

pred10 <- predict(bt_fit_1_39_res, new_data = reg_test) %>% 
  rename(bt_fit_1_39 = .pred) %>% 
  mutate(bt_fit_1_39wt = stack_coef %>% 
           filter(member == "bt_fit_1_39") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(bt_fit, bt_tictoc, bt_fit_1_35_wf, bt_fit_1_35_res, bt_fit_1_39_wf,
   bt_fit_1_39_res)

# Load cv
load("results/mlp_cv.rda")

# fit and predict
mlp_fit_1_05_wf <- mlp_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(mlp_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model05"))

mlp_fit_1_05_res <- fit(mlp_fit_1_05_wf, reg_train)

pred11 <- predict(mlp_fit_1_05_res, new_data = reg_test) %>% 
  rename(mlp_fit_1_05 = .pred) %>% 
  mutate(mlp_fit_1_05wt = stack_coef %>% 
           filter(member == "mlp_fit_1_05") %>% 
           select(weight) %>% 
           unlist())

# fit and predict
mlp_fit_1_10_wf <- mlp_fit %>%
  extract_workflow() %>% 
  finalize_workflow(show_best(mlp_fit, metric = "rmse", n = Inf) %>% 
                      filter(.config == "Preprocessor1_Model10"))

mlp_fit_1_10_res <- fit(mlp_fit_1_10_wf, reg_train)

pred12 <- predict(mlp_fit_1_10_res, new_data = reg_test) %>% 
  rename(mlp_fit_1_10 = .pred) %>% 
  mutate(mlp_fit_1_10wt = stack_coef %>% 
           filter(member == "mlp_fit_1_10") %>% 
           select(weight) %>% 
           unlist())

# Remove to free up memory  
rm(mlp_fit, mlp_tictoc, mlp_fit_1_05_wf, mlp_fit_1_05_res, mlp_fit_1_10_wf,
   mlp_fit_1_10_res)

pred1 <- pred1 %>% 
  select(elastic_psn_fit_1_05wt)

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
  cbind(pred11) %>% 
  cbind(pred12) %>% 
  mutate(y = elastic_psn_fit_1_05 * elastic_psn_fit_1_05wt +
           elastic_lm_int_fit_1_06 * elastic_lm_int_fit_1_06wt +
           elastic_lm_int_fit_1_11 * elastic_lm_int_fit_1_11wt +
           elastic_psn_int_fit_1_05 * elastic_psn_int_fit_1_05wt +
           elastic_nb_int_fit_1_05 * elastic_nb_int_fit_1_05wt + 	
           elastic_nb_int_fit_1_06 * elastic_nb_int_fit_1_06wt +
           gam_nb_fit_1_1 * gam_nb_fit_1_1wt +
           svm_rad_fit_1_6 * svm_rad_fit_1_6wt +
           bt_fit_1_35 * bt_fit_1_35wt +
           bt_fit_1_39 * bt_fit_1_39wt +
           mlp_fit_1_05 * mlp_fit_1_05wt +
           mlp_fit_1_10 * mlp_fit_1_10wt) %>% 
  restore_bounds()

# Save
save(ensemble_preds,
     file = "results/ensemble_preds.rda")