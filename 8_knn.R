# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, kknn)

# load saved objects from setup
load("results/modeling_objs.rda")

# set up knn model
knn_model <- nearest_neighbor(mode = "regression", 
                              neighbors = tune()) %>% 
  set_engine("kknn") 

## knn parameters
knn_params <- extract_parameter_set_dials(knn_model) %>% 
  update(neighbors = neighbors(range = c(1, 50)))
knn_grid <- grid_regular(knn_params, levels = 8)

# knn workflow
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(recipe_main)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("knn")

## fit knn
knn_fit <- knn_workflow %>% 
  tune_grid(reg_fold, grid = knn_grid,
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
knn_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(knn_fit, knn_tictoc, 
     file = "results/knn_cv.rda")

