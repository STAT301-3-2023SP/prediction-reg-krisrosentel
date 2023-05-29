# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, dbarts)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up bart model
bart_model <- 
  bart(
    trees = tune(),
    prior_terminal_node_coef = tune(),
    prior_terminal_node_expo = tune(),
    mode = "regression"
  ) %>% 
  set_engine("dbarts")
  
# bart parameters
bart_params <- extract_parameter_set_dials(bart_model) %>% 
  update(trees = trees(c(400, 1000)),
         prior_terminal_node_coef = prior_terminal_node_coef(c(.1,.99)),
         prior_terminal_node_expo = prior_terminal_node_expo(c(1, 5))) 
bart_grid <- grid_regular(bart_params, levels = c(3, 5, 5))

# bart workflow
bart_workflow <- workflow() %>% 
  add_model(bart_model) %>% 
  add_recipe(recipe_50) 

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("bart")

## fit bart
set.seed(41) # set seed 
bart_fit <- bart_workflow %>% 
  tune_grid(reg_fold, grid = bart_grid,
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
bart_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(bart_fit, bart_tictoc,  
     file = "results/bart_cv.rda")