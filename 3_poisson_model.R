# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, poissonreg)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up psn model
psn_model <- 
  poisson_reg() %>% 
  set_engine("glm")

# poisson workflow
psn_workflow <- workflow() %>% 
  add_model(psn_model) %>% 
  add_recipe(recipe_50)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("poisson")

## fit poisson 
psn_fit <- psn_workflow %>% 
  fit_resamples(resamples = reg_fold,
                control = control_resamples(save_pred = T,
                                            save_workflow = TRUE,
                                            parallel_over = "everything"),
                metrics = metric_set(rmse, rsq, smape))

# end parallel processing
stopCluster(cl)

# stop clock
toc(log = TRUE)
time_log <- tic.log(format = FALSE)

# save run time
psn_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(psn_fit, psn_tictoc,  
     file = "results/psn_cv.rda")