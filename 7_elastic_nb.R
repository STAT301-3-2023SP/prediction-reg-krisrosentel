# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, MASS, poissonreg)

# load saved objects from setup
load("results/modeling_objs.rda")

# estimate theta for neg bin using Poisson
reg_train <- read_csv("data/train.csv")
load("results/elastic_psn_cv.rda")

elastic_psn_results <- elastic_psn_fit %>% # fit Poisson model to train
  extract_workflow() %>% 
  finalize_workflow(select_best(elastic_psn_fit, metric = "rmse")) %>% 
  fit(., reg_train)

elastic_psn_pred <- predict(elastic_psn_results, new_data = reg_train)
theta <- theta.ml(y = reg_train$y, mu = elastic_psn_pred)

# set up elastic net model
elastic_nb_model <- linear_reg(mixture = tune(), 
                              penalty = tune()) %>% 
  set_engine("glmnet", family = MASS::negative.binomial(theta))

# elastic net parameters
elastic_params <- extract_parameter_set_dials(elastic_nb_model)
elastic_grid <- tibble(expand.grid(penalty = c(.05, .1, .5, 1, 5, 10, 20),
                                   mixture = c(0, .25, .5, .75, 1)))

# elastic workflow
elastic_nb_workflow <- workflow() %>% 
  add_model(elastic_nb_model) %>% 
  add_recipe(recipe_main)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("elastic net, neg. bin.")

## fit elastic net nb
elastic_nb_fit <- elastic_nb_workflow %>% 
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
elastic_nb_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(elastic_nb_fit, elastic_nb_tictoc,  
     file = "results/elastic_nb_cv.rda")