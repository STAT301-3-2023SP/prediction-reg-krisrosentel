# load libraries
library(pacman)
p_load(tidymodels, tidyverse, stacks, MASS, poissonreg, 
       mgcv, kknn, ranger, xgboost, earth, nnet, kernlab)

# deal with package conflicts
tidymodels_prefer()

# Load more
load("results/elastic_psn_cv.rda")

# Create data stack
reg_stack <- stacks() %>%
  add_candidates(elastic_psn_fit) 

# Remove to free up memory  
rm(elastic_psn_fit, elastic_psn_tictoc)

# Load more
load("results/elastic_lm_int_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
  add_candidates(elastic_lm_int_fit) 

# Remove to free up memory 
rm(elastic_lm_int_fit, elastic_lm_int_tictoc) 

# Load more
load("results/elastic_psn_int_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
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
load("results/gam_gaus_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
  add_candidates(gam_gaus_fit) 

# Remove to free up memory
rm(gam_gaus_fit, gam_gaus_tictoc)

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
load("results/bt_cv.rda")

# Add to data stack
reg_stack <- reg_stack %>%
  add_candidates(bt_fit) 

# Remove to free up memory 
rm(bt_fit, bt_tictoc)

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

# Fit the stack 
## penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.2, .3, .4, .5, .6, .7, .8, .9)


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
stack_coef <- collect_parameters(reg_blend, "elastic_psn_fit") %>% 
  full_join(collect_parameters(reg_blend, "elastic_lm_int_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "elastic_psn_int_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "elastic_nb_int_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "gam_gaus_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "gam_nb_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "svm_rad_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "bt_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "mlp_fit"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(reg_blend, "mars_fit"), 
            by = c("member", "coef")) %>% 
  filter(coef != 0) %>% 
  mutate(weight = coef / sum(coef))
  
# save
save(stack_coef,
     file = "results/ensemble_wts.rda")
save(reg_blend,
     file = "results/ensemble_blend.rda")

