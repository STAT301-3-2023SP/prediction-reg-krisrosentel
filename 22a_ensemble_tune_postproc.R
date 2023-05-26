# load libraries
library(pacman)
p_load(tidymodels, tidyverse, stacks, MASS, poissonreg, 
       mgcv, kknn, ranger, xgboost, earth, nnet, kernlab, baguette)

# deal with package conflicts
tidymodels_prefer()

# Load in
load("results/elastic_psn_int_cv.rda")

# Create stack
reg_stack <- stacks() %>%
  add_candidates(elastic_psn_int_fit)

# Remove to free up memory 
rm(elastic_psn_int_fit, elastic_psn_int_tictoc)

# Load more
load("results/elastic_nb_int_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
  add_candidates(elastic_nb_int_fit)

# Remove to free up memory 
rm(elastic_nb_int_fit, elastic_nb_int_tictoc)

# Load more
load("results/elastic_nb_pca_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
  add_candidates(elastic_nb_pca_fit)

# Remove to free up memory 
rm(elastic_nb_pca_fit, elastic_nb_pca_tictoc)

# Load more
load("results/gam_nb_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
  add_candidates(gam_nb_fit) 

# Remove to free up memory 
rm(gam_nb_fit, gam_nb_tictoc)

# Load more
load("results/svm_rad_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
  add_candidates(svm_rad_fit) 

# Remove to free up memory 
rm(svm_rad_fit, svm_rad_tictoc)

# Load more
load("results/bart_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
  add_candidates(bart_fit) 

# Remove to free up memory 
rm(bart_fit, bart_tictoc)

# Load more
load("results/mlp_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
  add_candidates(mlp_fit) 

# Remove to free up memory 
rm(mlp_fit, mlp_tictoc)

# Load more
load("results/mars_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
  add_candidates(mars_fit) 

# Remove to free up memory 
rm(mars_fit, mars_tictoc)

# write function to restore bounds of response variable 
restore_bounds <- function(x){ 
  x <- case_when(x < 1 ~ 1,
                 x > 100 ~ 100,
                 .default = x)
  x
}

# fix bounds in predictions
reg_stack <- reg_stack %>% 
  mutate_all(restore_bounds)

# view data stack
reg_stack

# Fit the stack 
## penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

## blend predictions using penalty defined above (tuning step, set seed)
reg_blend <-
  reg_stack %>%
  blend_predictions(penalty = blend_penalty,
                    metric = metric_set(rmse))

# remove to free up memory
rm(reg_stack)

# Explore the blended model stacks
autoplot(reg_blend)
autoplot(reg_blend, type = "weights")

# merge stacking coefs
stack_coef2 <- collect_parameters(reg_blend, "elastic_psn_int_fit") %>% 
  full_join(collect_parameters(reg_blend, "elastic_nb_int_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "elastic_nb_pca_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "gam_nb_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "svm_rad_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "bart_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "mlp_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "mars_fit"), 
            by = c("member", "coef")) %>% 
  filter(coef != 0) %>% 
  mutate(weight = coef / sum(coef))
  
# save
save(stack_coef2,
     file = "results/ensemble2_wts.rda")