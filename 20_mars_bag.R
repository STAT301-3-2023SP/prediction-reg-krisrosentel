# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, earth, baguette)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up mars model
mars_model <- bag_mars(
  mode = "regression",
  num_terms = tune(),
  prod_degree = tune()) %>%
  set_engine("earth", times = 30)

## mars parameters
mars_params <- extract_parameter_set_dials(mars_model) %>% 
  update(num_terms = num_terms(c(5, 62)),
         prod_degree = prod_degree(c(1, 3))) 
mars_grid <- grid_regular(mars_params, levels = c(20, 3))

# mars workflow
mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(recipe_50)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("mars")

## fit mars
set.seed(262) # set seed 
mars_fit <- mars_workflow %>% 
  tune_grid(reg_fold, grid = mars_grid,
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
mars_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(mars_fit, mars_tictoc, 
     file = "results/mars_cv.rda")

