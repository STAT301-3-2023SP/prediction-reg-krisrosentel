# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up elastic net model
elastic_lm_model <- linear_reg(mixture = tune(), 
                               penalty = tune()) %>% 
  set_engine("glmnet")

# elastic net parameters
elastic_params <- extract_parameter_set_dials(elastic_lm_model) %>% 
  update(penalty = penalty(range = c(-2.5, -.32)),
         mixture = mixture(range = c(0, .5)))
elastic_grid <- grid_regular(elastic_params, levels = 6)

# elastic workflow
elastic_lm_workflow <- workflow() %>% 
  add_model(elastic_lm_model) %>% 
  add_recipe(recipe_int)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("elastic net with interactions, linear")

## fit elastic net lm
elastic_lm_int_fit <- elastic_lm_workflow %>% 
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
elastic_lm_int_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(elastic_lm_int_fit, elastic_lm_int_tictoc,  
     file = "results/elastic_lm_int_cv.rda")