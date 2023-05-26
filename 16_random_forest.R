# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, ranger)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up random forest model
rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity")

## rf parameters
rf_params <- extract_parameter_set_dials(rf_model) %>% 
  update(min_n = min_n(c(2, 20)),
         mtry = mtry(c(5, 50))) 
rf_grid <- grid_regular(rf_params, levels = c(5, 5))

# rf workflow
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe_50)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("random forest")

## fit random forest
set.seed(322) # set seed 
rf_fit <- rf_workflow %>% 
  tune_grid(reg_fold, grid = rf_grid,
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
rf_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(rf_fit, rf_tictoc, 
     file = "results/rf_cv.rda")

