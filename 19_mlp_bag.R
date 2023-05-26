# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, nnet, baguette)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up mlp model
mlp_model <- bag_mlp(
  mode = "regression", 
  hidden_units = tune(),
  penalty = tune()) %>%
  set_engine("nnet", times = 80)

## mlp parameters
mlp_params <- extract_parameter_set_dials(mlp_model) %>% 
  update(hidden_units = hidden_units(c(1, 9)),
         penalty = penalty(c(1, 2.25))) 
mlp_grid <- grid_regular(mlp_params, levels = c(5, 6))

# mlp workflow
mlp_workflow <- workflow() %>% 
  add_model(mlp_model) %>% 
  add_recipe(recipe_50)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("mlp")

## fit mlp
mlp_fit <- mlp_workflow %>% 
  tune_grid(reg_fold, grid = mlp_grid,
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
mlp_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(mlp_fit, mlp_tictoc, 
     file = "results/mlp_cv.rda")

