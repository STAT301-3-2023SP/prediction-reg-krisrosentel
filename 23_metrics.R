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
models_table <- bind_rows(lm_tictoc,
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
  mutate("Run Time (min.)" = round(runtime / 60, 2)) %>% 
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
load("results/ensemble_wts.rda") # load in
ens_table1 <- stack_coef %>% 
  mutate(Penalty = case_when(!is.na(penalty.y) ~ as.character(round(penalty.y, 3)),
                             !is.na(penalty.x.x) ~ as.character(round(penalty.x.x, 3)),
                             !is.na(penalty.y.y) ~ as.character(round(penalty.y.y, 3)),
                             .default = "-"),
         Mixture = case_when(!is.na(mixture.y) ~ as.character(round(mixture.y, 3)),
                             !is.na(mixture) ~ as.character(round(mixture, 3)),
                             .default = "-"),
         Cost = case_when(!is.na(cost) ~ as.character(round(cost, 3)),
                             .default = "-"),
         Sigma = case_when(!is.na(rbf_sigma) ~ as.character(round(rbf_sigma, 3)),
                           .default = "-"),
         "Hide. Un." = case_when(!is.na(hidden_units) ~ as.character(hidden_units),
                                  .default = "-"),
         Bags = case_when(startsWith(member, "mlp") ~ "80",
                          startsWith(member, "mars") ~ "30",
                          .default = "-"),
         "Num. Terms" = case_when(!is.na(num_terms) ~ as.character(num_terms),
                                 .default = "-"),
         "Prod. Deg." = case_when(!is.na(prod_degree) ~ as.character(prod_degree),
                                  .default = "-"),
         "Stacking Coef." = round(coef, 7)) %>% 
  select(-member) %>% 
  cbind(Member = c("Elastic Net, Neg. Bin.", "Elastic Net, Neg. Bin. (PCA)", 
               "Elastic Net, Neg. Bin. (PCA)", "Elastic Net, Neg. Bin. (PCA)", 
               "Elastic Net, Neg. Bin. (PCA)", "Elastic Net, Neg. Bin. (PCA)", 
               "SVM, Radial", "MLP, Bagged", "MARS, Bagged", "MARS, Bagged")) %>% 
  arrange(desc(coef)) %>% 
  select(Member, Penalty, Mixture, Cost, Sigma, "Hide. Un.", Bags, "Num. Terms", "Prod. Deg.",
         "Stacking Coef.") 

## Runner Up Ensemble Plot
ens_plot1 <- ens_table1 %>% 
  rename(coef = "Stacking Coef.") %>% 
  cbind(lab = c("j", "i", "h", "g", "f", "e", "d", "c", "b", "a")) %>% 
  ggplot(aes(x = lab, y = coef, fill = Member)) +
           geom_bar(stat = "identity") + 
  coord_flip() +
  theme_minimal() +
  labs(title = "Ensemble Blend",
       y = "Stacking Coefficient",
       x = "Member Model") +
  theme(axis.text.y = element_blank()) +
  theme(plot.title = element_text(hjust = 0.5))

## Best Ensemble Table
load("results/ensemble2_wts.rda") # load in
ens_table2 <- stack_coef2 %>% 
  mutate(Penalty = case_when(!is.na(penalty.y) ~ as.character(round(penalty.y, 3)),
                             !is.na(penalty.x.x) ~ as.character(round(penalty.x.x, 3)),
                             !is.na(penalty.y.y) ~ as.character(round(penalty.y.y, 3)),
                             .default = "-"),
         Mixture = case_when(!is.na(mixture.y) ~ as.character(round(mixture.y, 3)),
                             !is.na(mixture) ~ as.character(round(mixture, 3)),
                             .default = "-"),
         Cost = case_when(!is.na(cost) ~ as.character(round(cost, 3)),
                          .default = "-"),
         Sigma = case_when(!is.na(rbf_sigma) ~ as.character(round(rbf_sigma, 3)),
                           .default = "-"),
         "Hide. Un." = case_when(!is.na(hidden_units) ~ as.character(hidden_units),
                                 .default = "-"),
         Bags = case_when(startsWith(member, "mlp") ~ "80",
                          startsWith(member, "mars") ~ "30",
                          .default = "-"),
         "Num. Terms" = case_when(!is.na(num_terms) ~ as.character(num_terms),
                                  .default = "-"),
         "Prod. Deg." = case_when(!is.na(prod_degree) ~ as.character(prod_degree),
                                  .default = "-"),
         "Stacking Coef." = round(coef, 7)) %>% 
  select(-member) %>% 
  cbind(Member = c("Elastic Net, Neg. Bin.", "Elastic Net, Neg. Bin. (PCA)", 
                   "Elastic Net, Neg. Bin. (PCA)", "SVM, Radial", 
                   "MLP, Bagged", "MARS, Bagged", "MARS, Bagged")) %>% 
  arrange(desc(coef)) %>% 
  select(Member, Penalty, Mixture, Cost, Sigma, "Hide. Un.", Bags, "Num. Terms", "Prod. Deg.",
         "Stacking Coef.") 

## Best Ensemble Plot
ens_plot2 <- ens_table2 %>% 
  rename(coef = "Stacking Coef.") %>% 
  cbind(lab = c("g", "f", "e", "d", "c", "b", "a")) %>% 
  ggplot(aes(x = lab, y = coef, fill = Member)) +
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_minimal() +
  labs(title = "Ensemble Blend",
       y = "Stacking Coefficient",
       x = "Member Model") +
  theme(axis.text.y = element_blank()) +
  theme(plot.title = element_text(hjust = 0.5))

# save
save(models_table, ens_plot1, ens_table1, ens_plot2, ens_table2,
     file = "results/memo_objs.rda")