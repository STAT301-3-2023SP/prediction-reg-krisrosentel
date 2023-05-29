# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up lm model
lm_model <- 
  linear_reg() %>% 
  set_engine("lm")

# lm workflow
lm_workflow <- workflow() %>% 
  add_model(lm_model) %>% 
  add_recipe(recipe_50)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("linear")

## fit lm
lm_fit <- lm_workflow %>% 
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
lm_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(lm_fit, lm_tictoc,  
     file = "results/lm_cv.rda")