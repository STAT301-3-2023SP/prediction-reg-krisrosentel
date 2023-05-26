# Load necessary libraries.
library(pacman)
p_load(tidymodels, tidyverse, yardstick, caret, MASS, poissonreg, mgcv, kknn, ranger,
       xgboost, earth, nnet, kernlab, baguette)

# deal with package conflicts
tidymodels_prefer()

# load model and preview results
load("results/lm_cv.rda")
lm_fit %>% 
  collect_metrics()
rm(lm_fit) # remove

# load model and preview results
load("results/psn_cv.rda")
psn_fit %>% 
  collect_metrics()
rm(psn_fit) # remove

# load model and preview results
load("results/nb_cv.rda")
nb_fit %>% 
  collect_metrics()
rm(nb_fit) # remove

# load model and preview results
load("results/elastic_lm_int_cv.rda")
elastic_lm_int_fit %>% 
  show_best(metric = "rmse")
rm(elastic_lm_int_fit) # remove

# load model and preview results
load("results/elastic_psn_int_cv.rda")
elastic_psn_int_fit %>% 
  show_best(metric = "rmse")
rm(elastic_psn_int_fit) # remove

# load model and preview results
load("results/elastic_nb_int_cv.rda")
elastic_nb_int_fit %>% 
  show_best(metric = "rmse")
rm(elastic_nb_int_fit) # remove

# load model and preview results
load("results/elastic_lm_pca_cv.rda")
elastic_lm_pca_fit %>% 
  show_best(metric = "rmse")
rm(elastic_lm_pca_fit) # remove

# load model and preview results
load("results/elastic_psn_pca_cv.rda")
elastic_psn_pca_fit %>% 
  show_best(metric = "rmse")
rm(elastic_psn_pca_fit) # remove

# load model and preview results
load("results/elastic_nb_pca_cv.rda")
elastic_nb_pca_fit %>% 
  show_best(metric = "rmse")
rm(elastic_nb_pca_fit) # remove

# load model and preview results
load("results/gam_gaus_cv.rda")
gam_gaus_fit %>% 
  show_best(metric = "rmse")
rm(gam_gaus_fit) # remove

# load model and preview results
load("results/gam_nb_cv.rda")
gam_nb_fit %>% 
  show_best(metric = "rmse")
rm(gam_nb_fit) # remove

# load model and preview results
load("results/knn_cv.rda")
knn_fit %>% 
  show_best(metric = "rmse")
rm(knn_fit) # remove

# load model and preview results
load("results/svm_poly_cv.rda")
svm_poly_fit %>% 
  show_best(metric = "rmse")
rm(svm_poly_fit) # remove

# load model and preview results
load("results/svm_rad_cv.rda")
svm_rad_fit %>% 
  show_best(metric = "rmse")
rm(svm_rad_fit) # remove

# load model and preview results
load("results/rf_cv.rda")
rf_fit %>% 
  show_best(metric = "rmse")
rm(rf_fit) # remove

# load model and preview results
load("results/bt_cv.rda")
bt_fit %>% 
  show_best(metric = "rmse")
rm(bt_fit) # remove

# load model and preview results
load("results/bart_cv.rda")
bart_fit %>% 
  show_best(metric = "rmse")
rm(bart_fit) # remove

# load model and preview results
load("results/mlp_cv.rda")
mlp_fit %>% 
  show_best(metric = "rmse")
rm(mlp_fit) # remove

# load model and preview results
load("results/mars_cv.rda")
mars_fit %>% 
  show_best(metric = "rmse")
rm(mars_fit) # remove

# load in ensemble results
load("results/ensemble_preds.rda") # post-processing applied after blend
load("results/ensemble_preds2.rda") # post-processing applied before blend

# Save predictions
runner_up <- ensemble_preds %>% select(id, y)
write.csv(runner_up, "submissions/pred_runnerup.csv", row.names = F)

best_model <- ensemble_preds2 %>% select(id, y)
write.csv(runner_up, "submissions/pred_best.csv", row.names = F)

# Build tables and plots for final memo
## Table of Model Tuning
model_tune <- bind_rows(lm_tictoc,
                        psn_tictoc,
                        nb_tictoc,
                        elastic_lm_int_tictoc,
                        elastic_psn_int_tictoc,
                        elastic_nb_int_tictoc,
                        elastic_lm_pca_tictoc,
                        elastic_psn_pca_tictoc,
                        elastic_nb_pca_tictoc,
                        gam_gaus_tictoc,
                        gam_nb_tictoc,
                        knn_tictoc,
                        svm_poly_tictoc,
                        svm_rad_tictoc,
                        rf_tictoc,
                        bt_tictoc,
                        bart_tictoc,
                        mlp_tictoc,
                        mars_tictoc) %>% 
  mutate("Run Time (min.)" = runtime / 60) %>% 
  cbind(Recipe = c("Recipe 1", "Recipe 1", "Recipe 1", "Recipe 2", "Recipe 2", "Recipe 2", 
                   "Recipe 3", "Recipe 3", "Recipe 3", "Recipe 1", "Recipe 1", "Recipe 1", 
                   "Recipe 2", "Recipe 2", "Recipe 1", "Recipe 1", "Recipe 1", "Recipe 1",
                   "Recipe 1")) %>% 
  cbind(Model = c("Linear Model", "Poisson", "Negative Binomial", "Elastic Net, Linear",
                  "Elastic Net, Poisson", "Elastic Net, Neg. Bin.", "Elastic Net, Linear",
                  "Elastic Net, Poisson", "Elastic Net, Neg. Bin.", 
                  "GAM, Gaussian", "GAM, Neg. Bin.", "KNN", "SVM, Polynomial", 
                  "SVM, Radial", "Random Forest", "Boosted Tree", "BART",
                  "MLP, Bagged", "MARS, Bagged")) %>% 
  select(Model, Recipe, "Run Time (min.)")

## Runner Up Ensemble Table


# save
save(model_plot, model_table, draw_confusion_matrix, conf_mat, auc,
     file = "results/exec_summary_objs.rda")