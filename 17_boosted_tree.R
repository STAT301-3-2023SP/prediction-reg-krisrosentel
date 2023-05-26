# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, xgboost)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up bt model
bt_model <- boost_tree(mode = "regression",
                       min_n = tune(),
                       mtry = tune(),
                       learn_rate = tune()) %>% 
  set_engine("xgboost")

## bt parameters
bt_params <- extract_parameter_set_dials(bt_model) %>% 
  update(min_n = min_n(c(2, 50)),
         mtry = mtry(c(5, 50)),
         learn_rate = learn_rate(c(-.8, -.3)))
bt_grid <- grid_regular(bt_params, levels = 4)

# bt workflow
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(recipe_50)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("boosted tree")

## fit boosted tree
set.seed(211) # set seed 
bt_fit <- bt_workflow %>% 
  tune_grid(reg_fold, grid = bt_grid,
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
bt_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(bt_fit, bt_tictoc, 
     file = "results/bt_cv.rda")

