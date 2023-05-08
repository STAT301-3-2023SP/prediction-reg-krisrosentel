# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, mgcv)

# load saved objects from setup
load("results/modeling_objs.rda")

# set up gam
gam_model <- gen_additive_mod(
  mode = "regression",
  select_features = tune(),
  adjust_deg_free = tune()) %>% 
  set_engine("mgcv", family = poisson(link = log))

# gam parameters
gam_params <- extract_parameter_set_dials(gam_model)
gam_grid <- grid_regular(gam_params, levels = c(2, 6))

# gam workflow
gam_workflow <- workflow() %>% 
  add_model(gam_model) %>% 
  add_recipe(recipe_interact)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("gam")

## fit gam
gam_fit <- gam_workflow %>% 
  tune_grid(reg_fold, grid = gam_grid,
            control = control_grid(save_pred = TRUE, 
                                   save_workflow = TRUE,
                                   parallel_over = "everything"))

# end parallel processing
stopCluster(cl)

# stop clock
toc(log = TRUE)
time_log <- tic.log(format = FALSE)

# save run time
gam_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(gam_fit, gam_tictoc, 
     file = "results/gam_cv.rda")

