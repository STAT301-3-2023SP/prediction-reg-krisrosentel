# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, mgcv)

# load saved objects from setup
load("results/modeling_objs.rda")

# define gam formula 
gam_formula <- 
  as.formula(str_c("y", "~", 
                   str_c((str_c("s(", best_pred50, ")")), collapse = " + "), sep = " "))

# set up gam
gam_nb_model <- gen_additive_mod(
  mode = "regression",
  select_features = TRUE,
  adjust_deg_free = tune()) %>% 
  set_engine("mgcv", family = nb(2.715))

# gam parameters
gam_params <- extract_parameter_set_dials(gam_nb_model)
gam_grid <- grid_regular(gam_params, levels = 5)

# gam workflow
gam_nb_workflow <- workflow() %>% 
  add_model(gam_nb_model, formula = gam_formula) %>% 
  add_recipe(recipe_gam)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("gam, neg. bin.")

## fit nb gam
gam_nb_fit <- gam_nb_workflow %>% 
  tune_grid(reg_fold, grid = gam_grid,
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
gam_nb_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(gam_nb_fit, gam_nb_tictoc, 
     file = "results/gam_nb_cv.rda")

