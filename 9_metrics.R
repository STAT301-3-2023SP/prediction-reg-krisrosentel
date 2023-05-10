# Load necessary libraries.
library(pacman)
p_load(tidymodels, tidyverse, yardstick, caret, poissonreg)

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
load("results/gam_gaus_cv.rda")

lm_fit %>% 
  collect_metrics()

psn_fit %>% 
  collect_metrics()

nb_fit %>% 
  collect_metrics()

elastic_lm_fit %>% 
  show_best(metric = "rmse")

elastic_psn_fit %>% 
  show_best(metric = "rmse")

gam_gaus_fit %>% 
  show_best(metric = "rmse")

# finalize best model- elastic net, linear
elastic_lm_tuned_wf <- elastic_lm_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(elastic_lm_fit, metric = "rmse"))

# fit
elastic_lm_results <- fit(elastic_lm_tuned_wf, reg_train)

# finalize best model- elastic net, poisson
elastic_psn_tuned_wf <- elastic_psn_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(elastic_psn_fit, metric = "rmse"))

# fit
elastic_psn_results <- fit(elastic_psn_tuned_wf, reg_train)

# finalize best model- gam, gaussian
gam_gaus_tuned_wf <- gam_gaus_fit %>%
  extract_workflow() %>% 
  finalize_workflow(select_best(gam_gaus_fit, metric = "rmse"))

# fit
gam_gaus_results <- fit(gam_gaus_tuned_wf, reg_train)

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
predictions3 <- predict(elastic_psn_results, new_data = reg_test) %>% 
  rename(y = .pred) %>%  
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions_bounded3 <- predict(elastic_psn_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  restore_bounds() %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

# calc predictions - gam
predictions4 <- predict(gam_gaus_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)

predictions_bounded4 <- predict(gam_gaus_results, new_data = reg_test) %>% 
  rename(y = .pred) %>% 
  restore_bounds() %>% 
  bind_cols(reg_test %>% select(id)) %>% 
  select(id, y)


# save
write.csv(predictions2, "submissions/pred2_050723.csv", row.names = F)
write.csv(predictions_bounded2, "submissions/pred_bnd2_050723.csv", row.names = F)

write.csv(predictions3, "submissions/pred3_050723.csv", row.names = F)
write.csv(predictions_bounded3, "submissions/pred_bnd3_050723.csv", row.names = F)  

write.csv(predictions4, "submissions/pred4_050923.csv", row.names = F)
write.csv(predictions_bounded4, "submissions/pred_bnd4_050923.csv", row.names = F)  

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

# finalize best model- elastic net
elastic_tuned_workflow <- elastic_workflow %>% 
  finalize_workflow(select_best(elastic_tuned, metric = "roc_auc"))

# fit
elastic_results <- fit(elastic_tuned_workflow, fire_train)

# Calculate AUC
auc <- predict(elastic_results, new_data = fire_test, type = "prob") %>% 
  bind_cols(fire_test %>% select(wlf)) %>% 
  roc_auc(wlf, .pred_yes) %>% 
  select(.estimate) %>% 
  unlist()

# confusion matrix
predictions <- predict(elastic_results, new_data = fire_test) %>% 
  bind_cols(fire_test %>% select(wlf)) %>% 
  rename(pred_wlf = .pred_class)

conf_mat <- confusionMatrix(data = predictions$pred_wlf, 
                            reference = predictions$wlf)

# confusion matrix formatting function
draw_confusion_matrix <- function(cm, auc) {
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Fire', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'No Fire', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Fire', cex=1.2, srt=90)
  text(140, 335, 'No Fire', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.8, font=2, col='white')
  text(195, 335, res[2], cex=1.8, font=2, col='white')
  text(295, 400, res[3], cex=1.8, font=2, col='white')
  text(295, 335, res[4], cex=1.8, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "METRICS", xaxt='n', yaxt='n')
  text(15, 65, names(cm$byClass[1]), cex=1.4, font=2)
  text(15, 45, round(as.numeric(cm$byClass[1]), 3), cex=1.4)
  text(40, 65, names(cm$byClass[2]), cex=1.4, font=2)
  text(40, 45, round(as.numeric(cm$byClass[2]), 3), cex=1.4)
  text(63, 65, names(cm$overall[1]), cex=1.4, font=2)
  text(63, 45, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(85, 65, "ROC AUC", cex=1.3, font=2)
  text(85, 45, round(auc, 3), cex=1.4)
}  

metrics_plot <- draw_confusion_matrix(conf_mat, auc)

# save
save(model_plot, model_table, draw_confusion_matrix, conf_mat, auc,
     file = "results/exec_summary_objs.rda")