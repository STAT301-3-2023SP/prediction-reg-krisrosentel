# Load necessary libraries.
library(pacman)
p_load(tidymodels, tidyverse, yardstick, caret, MASS, poissonreg, mgcv, kknn, ranger,
       xgboost, earth, nnet, kernlab)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup and tuning
reg_train <- read_csv("data/train.csv") 
reg_test <- read_csv("data/test.csv")
load("results/lm_cv.rda")
load("results/psn_cv.rda")
load("results/nb_cv.rda")
load("results/elastic_lm_cv.rda")
load("results/elastic_psn_cv.rda")
load("results/elastic_nb_cv.rda")
load("results/knn_cv.rda")
load("results/rf_cv.rda")
load("results/bt_cv.rda")
load("results/svm_poly_cv.rda")
load("results/svm_rad_cv.rda")
load("results/mlp_cv.rda")
load("results/mars_cv.rda")
load("results/gam_gaus_cv.rda")
load("results/gam_nb_cv.rda")
load("results/elastic_lm_int_cv.rda")
load("results/elastic_psn_int_cv.rda")
load("results/elastic_nb_int_cv.rda")
load("results/ensemble_preds.rda")
load("results/gam_nb_modelres.rda")

lm_fit %>% 
  collect_metrics()

psn_fit %>% 
  collect_metrics()

nb_fit %>% 
  collect_metrics()

elastic_lm_fit %>% 
  show_best(metric = "rmse")

elastic_lm_fit %>% 

elastic_psn_fit %>% 
  show_best(metric = "rmse", 25) %>% 
  print(n = 25)

elastic_psn_fit %>% 
  autoplot()

elastic_nb_fit %>% 
  show_best(metric = "rmse")

gam_gaus_fit %>% 
  show_best(metric = "rmse")

gam_nb_fit %>% 
  show_best(metric = "rmse") 

knn_fit %>% 
  show_best(metric = "smape")

rf_fit %>% 
  autoplot(metric = "rmse")

bt_fit %>% 
  show_best(metric = "rmse")

svm_poly_fit %>% 
  show_best(metric = "rmse")

svm_rad_fit %>% 
  autoplot(metric = "rmse")

svm_rad_fit %>% 
  autoplot(metric = "rmse")

mlp_fit %>% 
  autoplot(metric = "rmse")

mars_fit %>% 
  autoplot(metric = "rmse")

elastic_lm_int_fit %>% 
  show_best(metric = "rmse")

elastic_nb_int_fit %>% 
  show_best(metric = "rmse")

elastic_psn_int_fit %>% 
  show_best(metric = "rmse")

# finalize best model- elastic net, linear
elastic_lm_tuned_wf <- elastic_lm_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(elastic_lm_fit, metric = "rmse"))

# fit
elastic_lm_results <- fit(elastic_lm_tuned_wf, reg_train)

# finalize best model- elastic net, poisson
elastic_psn_int_tuned_wf <- elastic_psn_int_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(elastic_psn_int_fit, metric = "rmse"))

# fit
elastic_psn_int_results <- fit(elastic_psn_int_tuned_wf, reg_train)

# finalize best model- elastic net, nb
elastic_nb_int_tuned_wf <- elastic_nb_int_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(elastic_nb_int_fit, metric = "rmse"))

# fit
elastic_nb_int_results <- fit(elastic_nb_int_tuned_wf, reg_train)

# finalize best model- gam, gaussian
gam_gaus_tuned_wf <- gam_gaus_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(gam_gaus_fit, metric = "rmse"))

# fit
gam_gaus_results <- fit(gam_gaus_tuned_wf, reg_train)

# finalize best model- gam, nb
gam_nb_tuned_wf <- gam_nb_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(gam_nb_fit, metric = "rmse"))

# fit
gam_nb_results <- fit(gam_nb_tuned_wf, reg_train)

# finalize best model- knn
knn_tuned_wf <- knn_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(knn_fit, metric = "rmse"))

# fit
knn_results <- fit(knn_tuned_wf, reg_train)

# finalize best model- rf
rf_tuned_wf <- rf_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(rf_fit, metric = "rmse"))

# fit
rf_results <- fit(rf_tuned_wf, reg_train)

# finalize best model- bt
bt_tuned_wf <- bt_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(bt_fit, metric = "rmse"))

# fit
bt_results <- fit(bt_tuned_wf, reg_train)

# finalize best model- svm poly
svm_poly_tuned_wf <- svm_poly_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(svm_poly_fit, metric = "rmse"))

# fit
svm_poly_results <- fit(svm_poly_tuned_wf, reg_train)

# finalize best model- svm rad
svm_rad_tuned_wf <- svm_rad_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(svm_rad_fit, metric = "rmse"))

# fit
svm_rad_results <- fit(svm_rad_tuned_wf, reg_train)

# finalize best model- mars rad
mars_tuned_wf <- mars_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(mars_fit, metric = "rmse"))

# fit
mars_results <- fit(mars_tuned_wf, reg_train)

# write function to restore bounds of response variable 
restore_bounds <- function(x){ 
  x <- x %>% mutate(y = case_when(y < 1 ~ 1,
                                  y > 100 ~ 100,
                                  .default = y))
}

# calc predictions - elastic lm
predictions2 <- predict(elastic_lm_results, new_data = reg_test) %>% 
  rename(y = .pred) %>%  
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions_bounded2 <- predict(elastic_lm_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  restore_bounds() %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

# calc predictions - elastic psn
predictions8 <- predict(elastic_nb_results, new_data = reg_test) %>% 
  rename(y = .pred) %>%  
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions_bounded8 <- predict(elastic_nb_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  restore_bounds() %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

# calc predictions - gam, gaus
predictions4 <- predict(gam_gaus_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions_bounded4 <- predict(gam_gaus_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  restore_bounds() %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

# calc predictions - gam, nb
predictions5 <- predict(gam_nb_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions_bounded5 <- predict(gam_nb_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  restore_bounds() %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

# save nb gam model results because it takes a long time
save(gam_nb_results, predictions5, predictions_bounded5, 
     file = "results/gam_nb_modelres.rda")

# calc predictions - knn
predictions11 <- predict(knn_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions_bounded11 <- predict(knn_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  restore_bounds() %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

# calc predictions - rf
predictions12 <- predict(rf_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions_bounded12 <- predict(rf_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  restore_bounds() %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

# calc predictions - bt
predictions14 <- predict(bt_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions_bounded14 <- predict(bt_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  restore_bounds() %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions19 <- predict(svm_poly_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions22 <- predict(svm_rad_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions20 <- predict(mars_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions23 <- predict(elastic_nb_int_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions24 <- ensemble_preds %>% select(id, y)

# save
write.csv(predictions2, "submissions/pred2_050723.csv", row.names = F)
write.csv(predictions_bounded2, "submissions/pred_bnd2_050723.csv", row.names = F)

write.csv(predictions3, "submissions/pred3_050723.csv", row.names = F)
write.csv(predictions_bounded3, "submissions/pred_bnd3_050723.csv", row.names = F)  

write.csv(predictions4, "submissions/pred4_050923.csv", row.names = F)
write.csv(predictions_bounded4, "submissions/pred_bnd4_050923.csv", row.names = F)  

write.csv(predictions5, "submissions/pred5_051223.csv", row.names = F)
write.csv(predictions_bounded5, "submissions/pred_bnd5_051223.csv", row.names = F)  

write.csv(predictions8, "submissions/pred8_051323.csv", row.names = F)
write.csv(predictions_bounded8, "submissions/pred_bnd8_051323.csv", row.names = F)  

write.csv(predictions11, "submissions/pred11_051323.csv", row.names = F)
write.csv(predictions_bounded11, "submissions/pred_bnd11_051323.csv", row.names = F)  
write.csv(predictions12, "submissions/pred12_051323.csv", row.names = F)
write.csv(predictions_bounded12, "submissions/pred_bnd12_051323.csv", row.names = F)  

write.csv(predictions14, "submissions/pred14_051423.csv", row.names = F)

write.csv(predictions19, "submissions/pred19_051723.csv", row.names = F)

write.csv(predictions23, "submissions/pred23_051823.csv", row.names = F)

write.csv(predictions24, "submissions/pred_bnd24_052123.csv", row.names = F)

########### Scratch from previous labs- save for later
gam_gaus_fit %>% 
  filter(id == "Repeat3" & id2 == "Fold5") %>% 
  select(.notes) %>% 
  unlist()

# load saved objects from setup and tuning
load("results/data_split.rda")
load("results/elastic_workflow.rda")
load("results/elastic_cv.rda")
load("results/knn_cv.rda")
load("results/rf_cv.rda")
load("results/bt_cv.rda")
load("results/svm_poly_cv.rda")
load("results/svm_rad_cv.rda")
load("results/mlp_cv.rda")
load("results/mars_cv.rda")
load("results/null_cv.rda")

# combine results into single set
model_set <- as_workflow_set(
  "elastic_net" = elastic_tuned,
  "knn" = knn_tuned,
  "rand_forest" = rf_tuned,
  "boosted_tree" = bt_tuned,
  "svm_poly" = svm_poly_tuned,
  "svm_rbf" = svm_rad_tuned,
  "mlp" = mlp_tuned,
  "mars" = mars_tuned,
  "null" = null_fit
)

# plot results
model_plot <- model_set %>% 
  autoplot(metric = "roc_auc", select_best = TRUE) +
  theme_minimal() + 
  guides(shape = FALSE) +
  scale_color_discrete(breaks = c('logistic_reg', 'svm_poly', 'mlp',
                                  'svm_rbf', 'boost_tree', 'rand_forest',
                                  'mars', 'nearest_neighbor', 'null_model'),
                       labels = c('elastic_net', 'svm_poly', 'mlp',
                                  'svm_rbf', 'boost_tree', 'rand_forest',
                                  'mars', 'nearest_neighbor', 'null_model')) +
  labs(y = "ROC AUC",
       title = "Performance for Best Model of Each Type") +
  theme(axis.text.x = element_blank(),
        legend.title.align=0.5,
        plot.title = element_text(hjust = 0.5))

# table of results
model_table <- model_set %>% 
  group_by(wflow_id) %>% 
  mutate(best = map(result, show_best, metric = "roc_auc")) %>% 
  select(best) %>% 
  unnest(cols = c(best)) %>% 
  slice_max(mean) %>% 
  arrange(desc(mean))

# computation times
model_times <- bind_rows(elastic_tictoc,
                         svm_poly_tictoc,
                         mlp_tictoc,
                         svm_rad_tictoc,
                         bt_tictoc,
                         rf_tictoc,
                         mars_tictoc,
                         knn_tictoc,
                         null_tictoc) %>% 
  mutate(runtime = runtime / 60) %>% 
  mutate(model = ifelse(model == "multilayer perception", 
                        "multilayer perceptron", model))

# add to table
model_table <- model_table %>% cbind(model_times)


# save
save(model_plot, model_table, draw_confusion_matrix, conf_mat, auc,
     file = "results/exec_summary_objs.rda")