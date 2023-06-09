# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, MASS, poissonreg)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# estimate theta for neg bin using Poisson
reg_train <- read_csv("data/train.csv")
load("results/psn_cv.rda")

psn_results <- psn_fit %>% # fit Poisson model to train
  extract_workflow() %>% 
  finalize_workflow(select_best(psn_fit, metric = "rmse")) %>% 
  fit(., reg_train)

psn_pred <- predict(psn_results, new_data = reg_train)
theta <- theta.ml(y = reg_train$y, mu = psn_pred)

# set up negbin model
nb_model <- 
  linear_reg() %>% 
  set_engine("glm", family = MASS::negative.binomial(theta)) 

# nb workflow
nb_workflow <- workflow() %>% 
  add_model(nb_model) %>% 
  add_recipe(recipe_50)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("neg. bin.")

## fit nb
nb_fit <- nb_workflow %>% 
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
nb_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(nb_fit, nb_tictoc,  
     file = "results/nb_cv.rda")