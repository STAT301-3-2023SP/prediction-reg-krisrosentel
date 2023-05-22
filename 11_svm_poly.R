# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, kernlab)

# load saved objects from setup
load("results/modeling_objs.rda")

# set up svm model
svm_poly_model <- svm_poly(
  mode = "regression",
  cost = tune(),
  degree = tune()) %>%
  set_engine("kernlab")

## svm parameters
svm_poly_params <- extract_parameter_set_dials(svm_poly_model) %>% 
  update(cost = cost(c(-2.5, 5.5))) 
svm_poly_grid <- grid_regular(svm_poly_params, levels = c(5, 3)) 

# svm workflow
svm_poly_workflow <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(recipe_main)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("svm poly")

## fit svm
svm_poly_fit <- svm_poly_workflow %>% 
  tune_grid(reg_fold, grid = svm_poly_grid,
            control = control_grid(save_pred = TRUE, 
                                   save_workflow = TRUE,
                                   parallel_over = "everything"),
            metrics = metric_set(rmse, rsq, smape))

# end parallel processing
stopCluster(cl)

# stop clock
toc(log = TRUE)
time_log <- tic.log(format = FALSE)

# save run time
svm_poly_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(svm_poly_fit, svm_poly_tictoc, 
     file = "results/svm_poly_cv.rda")

