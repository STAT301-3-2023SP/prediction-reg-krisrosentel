# Load package(s)
library(pacman)
p_load(tidymodels, tidyverse, skimr, vip, psych)

# handle common conflicts
tidymodels_prefer()

# load data
reg_train <- read_csv("data/train.csv") 

# plot y- looks to be count data and positively skewed. will want to run Poisson model and 
# elastic net Poisson 
reg_train %>% 
  ggplot(aes(x = y)) +
  geom_histogram()

# run an initial random forest on subset of data to select 250 most important predictors 
## create subset of data - 20% of data
set.seed(212) # set seed
reg_split <- initial_split(reg_train, prop = 0.2, strata = y)
reg_include <- training(reg_split) # subset to determine important predictors

##  make list of vars  with any missing
miss_screen <- reg_include %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

## recipe 
recipe_screen <- recipe(y ~ ., data = reg_include) %>%
  step_rm(id) %>% 
  step_impute_knn(miss_screen) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())  # center and scale

## prep and bake
recipe_screen %>% 
  prep() %>% 
  bake(new_data = reg_include) %>% 
  head(15) 

## set up rf model
rf_screen <- rand_forest(mode = "regression") %>% 
  set_engine("ranger", importance = "impurity")

## rf workflow
rf_screen_workflow <- workflow() %>% 
  add_model(rf_screen) %>% 
  add_recipe(recipe_screen)

## fit model
set.seed(2)
screen_fit <- fit(rf_screen_workflow, reg_include) 

## Create list of 250 best predictors from initial rf
best_pred <- screen_fit %>% 
  extract_fit_parsnip() %>% 
  vip::vi() %>% 
  head(250) %>% 
  select(Variable) %>% 
  unlist() %>% 
  unname()

# Create lists of variables for recipe steps
## list of highly skewed predictors for  YeoJohnson
skew_pred <- reg_train %>% 
  select(best_pred) %>% 
  describe() %>% 
  filter(skew > 3 | skew < -3) %>% 
  select(vars, skew, min, max) %>% 
  arrange(desc(skew)) %>% 
  rownames()

## list of highly correlated predictors for step_PCA
pca_pred <- reg_train %>% 
  drop_na() %>% 
  select(best_pred) %>% 
  cor() %>% 
  as.table() %>% 
  as.data.frame() %>% 
  filter(Var1 != Var2) %>% 
  mutate(Freq = abs(Freq)) %>% 
  arrange(desc(Freq)) %>% 
  filter(Freq > .9) %>% 
  select(Var1) %>% 
  unlist() %>% 
  as.character() %>% 
  unname() %>% 
  unique()

## list of vars for interactions (5 most important in rf that are not in PCA)
int_pred <- best_pred %>% 
  subset(!best_pred %in% pca_pred) %>% 
  head(5)

## list of vars with any missingness
miss_train <- reg_train %>% # missing in training
  select(best_pred) %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

reg_test <- read_csv("data/test.csv") # load in test data

miss_test <- reg_test %>% # missing in testing
  select(best_pred) %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

miss_test %>% # vars w/ missing in test but not train
  subset(!miss_test %in% miss_train) 

# create list of vars missing in either test or train
miss_combo <- c(miss_train, miss_test) %>% 
  unique()

# set up folds
set.seed(4006)
reg_fold <- vfold_cv(reg_train, v = 5, repeats = 3)

# set up main recipe
recipe_main <- recipe(y ~ ., data = reg_train) %>%
  step_rm(all_predictors(), -best_pred) %>% # remove all but the best 250 predictors
  step_normalize(all_predictors()) %>% # normalize before knn imputation
  step_impute_knn(miss_combo) %>% # impute using knn
  step_YeoJohnson(skew_pred) %>% # transform highly skewed predictors
  step_normalize(all_predictors()) %>% # normalize again after transformations, imputation
  step_pca(pca_pred, threshold = .9) %>% # conduct PCA with highly correlated vars 
  step_nzv(all_predictors(), unique_cut = 1) %>% # remove predictors with near zero var 
  step_normalize(all_predictors()) # normalize again after conducting PCA 

# prep and bake
recipe_main %>% 
  prep() %>% 
  bake(new_data = reg_train) %>% 
  head(15) 

# add recipe with interactions
recipe_interact <- recipe_main %>% 
  step_interact(~c(starts_with("PC"), int_pred)^2) 

# prep and bake
recipe_interact %>% 
  prep() %>% 
  bake(new_data = reg_train) %>% 
  head(15)

save(best_pred, miss_combo, skew_pred, pca_pred, int_pred, 
     recipe_main, recipe_interact, reg_fold,
     file = "results/modeling_objs.rda")