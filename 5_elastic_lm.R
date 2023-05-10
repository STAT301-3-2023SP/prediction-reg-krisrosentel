# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc)

# load saved objects from setup
load("results/modeling_objs.rda")

# set up elastic net model
elastic_lm_model <- linear_reg(mixture = tune(), 
                              penalty = tune()) %>% 
  set_engine("glmnet")

# elastic net parameters
elastic_params <- extract_parameter_set_dials(elastic_lm_model)
elastic_grid <- tibble(expand.grid(penalty = c(.05, .1, .5, 1, 5, 10, 20),
                                   mixture = c(0, .25, .5, .75, 1)))

# elastic workflow
elastic_lm_workflow <- workflow() %>% 
  add_model(elastic_lm_model) %>% 
  add_recipe(recipe_main)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("elastic net, linear")

## fit elastic net lm
elastic_lm_fit <- elastic_lm_workflow %>% 
  tune_grid(reg_fold, grid = elastic_grid,
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
elastic_lm_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(elastic_lm_fit, elastic_lm_tictoc,  
     file = "results/elastic_lm_cv.rda")