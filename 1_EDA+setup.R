# Load package(s)
library(pacman)
p_load(tidymodels, tidyverse, stringr, skimr, vip, psych)

# handle common conflicts
tidymodels_prefer()

# load data
reg_train <- read_csv("data/train.csv") 

# plot y- looks to be count data and positively skewed. will want to run Poisson model and 
# elastic net Poisson. Also neg binomial if overdispersed 
reg_train %>% 
  ggplot(aes(x = y)) +
  geom_histogram()

# run an initial random forest on subset of data to select most important predictors 
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
# recipe_screen %>% 
  # prep() %>% 
  # bake(new_data = NULL) %>% 
  # head(15) 

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

## Create list of 300 best predictors from initial rf
best_pred <- screen_fit %>% 
  extract_fit_parsnip() %>% 
  vip::vi() %>% 
  head(300) %>% 
  select(Variable) %>% 
  unlist() %>% 
  unname()

## Create list of 50 best predictors from initial rf aftter dropping high corr
best_pred50 <- screen_fit %>% 
  extract_fit_parsnip() %>% 
  vip::vi() %>% 
  head(93) %>% 
  filter(!(Variable %in% c("x724", "x651", "x591", "x102", "x472", "x753",
                      "x488", "x756", "x755", "x572", "x366", "x687", "x391", 
                      "x147", "x378", "x569", "x239", "x118", "x609", "x548",
                      "x430", "x244", "x619", "x231", "x670", "x515", "x350",
                      "x748", "x638", "x669", "x203", "x342", "x420", "x683",
                      "x499", "x355", "x005", "x563", "x317", 
                      "x731", "x165", "x220", "x487"))) %>% 
  select(Variable) %>% 
  unlist() %>% 
  unname()

# examine highly correlated predictors among best 50 
# run, update selection to rm, and iterate until no very high correlation
reg_train %>% 
  drop_na() %>% 
  select(best_pred50) %>% 
  cor() %>% 
  as.table() %>% 
  as.data.frame() %>% 
  filter(Var1 != Var2) %>% 
  mutate(Freq = abs(Freq)) %>%
  arrange(desc(Freq)) %>% 
  filter(Freq > .9) %>%
  filter(seq_len(nrow(.)) %% 2 == 0) %>% 
  filter(!Var1 %in% Var2) %>% 
  select(Var1) %>% 
  unlist() %>% 
  unique()

# Create lists of variables for recipe steps
## list of highly skewed predictors for  YeoJohnson
skew_pred <- reg_train %>% 
  select(all_of(best_pred)) %>% 
  describe() %>% 
  filter(skew > 1 | skew < -1) %>% 
  select(vars, skew, min, max) %>% 
  arrange(desc(skew)) %>% 
  rownames()

## list of highly skewed predictors for YeoJohnson 
skew_pred50 <- reg_train %>% 
  select(all_of(best_pred50)) %>% 
  describe() %>% 
  filter(skew > 3 | skew < -3) %>% 
  select(vars, skew, min, max) %>% 
  arrange(desc(skew)) %>% 
  rownames()

## list of vars with any missingness for PCA
miss_train <- reg_train %>% # missing in training
  select(all_of(best_pred)) %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

## list of vars with any missingness for main
miss_train50 <- reg_train %>% # missing in training
  select(all_of(best_pred50)) %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

reg_test <- read_csv("data/test.csv") # load in test data

## list of vars with any missingness for PCA in test
miss_test <- reg_test %>% # missing in testing
  select(best_pred) %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

miss_test %>% # vars w/ missing in test but not train
  subset(!miss_test %in% miss_train) 

miss_combo <- c(miss_train, miss_test) %>% # missing in either test or train
  unique()

## list of vars with any missingness for main in test
miss_test50 <- reg_test %>% # missing in testing
  select(best_pred50) %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

miss_test50 %>% # vars w/ missing in test but not train
  subset(!miss_test50 %in% miss_train50) 

miss_combo50 <- c(miss_train50, miss_test50) %>% # missing in either test or train
  unique()

# set up folds
set.seed(280)
reg_fold <- vfold_cv(reg_train, v = 5, repeats = 3)

# set up main recipe
recipe_50 <- recipe(y ~ ., data = reg_train) %>% 
  step_rm(all_predictors(), -best_pred50) %>% # recipe
  step_normalize(all_predictors()) %>% # normalize before knn imputation
  step_impute_knn(miss_combo50) %>% # impute using knn
  step_YeoJohnson(skew_pred50) %>% # transform highly skewed predictors
  step_normalize(all_predictors()) # normalize again after transformations and imputation

# prep and bake
# recipe_50 %>% 
  # prep() %>% 
  # bake(new_data = NULL) %>% 
  # head(15)  

# set up interaction recipe
recipe_int <- recipe_50 %>% 
  step_interact(~ x631:x702 + x604:x702 + x073:x702 + x105:x702 + x105:x114 +
                  x114:x146 + x026:x096 + x307:x631 + x661:x702 + x114:x631 + x073:x704 +
                  x105:x253 + x589:x664 + x543:x589 + x073:x105 + x253:x680 + x307:x416 +
                  x026:x664 + x092:x636 + x425:x685 + x716:x721 + x653:x680 + x135:x307 +
                  x653:x721 + x135:x661 + x093:x416 + x307:x447 + x096:x135 + x105:x653 +
                  x604:x664 + x307:x427 + x105:x307 + x096:x457 + x685:x704 + x127:x693 +
                  x662:x749 + x093:x631 + x073:x146 + x080:x696 + x265:x661 + x514:x696 +
                  x111:x416 + x105:x111 + x114:x253 + x092:x274 + x135:x416 + x127:x653 +
                  x589:x704 + x026:x661 + x105:x364)

# prep and bake
# recipe_int %>% 
  # prep() %>% 
  # bake(new_data = NULL) %>% 
  # head(15)  

# set up PCA recipe
recipe_pca <- recipe(y ~ ., data = reg_train) %>%
  step_rm(all_predictors(), -best_pred) %>% # remove all but the best 300 predictors
  step_YeoJohnson(skew_pred) %>% # transform predictors before impute and PCA
  step_normalize(all_predictors()) %>% # normalize before knn imputation
  step_impute_knn(miss_combo) %>% # impute using knn
  step_pca(all_predictors(), -best_pred50, num_comp = 80) %>% # conduct PCA since some vars are highly correlated
  step_interact(~ x631:x702 + x604:x702 + x073:x702 + x105:x702 + x105:x114 + # add interactions
                  x114:x146 + x026:x096 + x307:x631 + x661:x702 + x114:x631 + x073:x704 +
                  x105:x253 + x589:x664 + x543:x589 + x073:x105 + x253:x680 + x307:x416 +
                  x026:x664 + x092:x636 + x425:x685 + x716:x721 + x653:x680 + x135:x307 +
                  x653:x721 + x135:x661 + x093:x416 + x307:x447 + x096:x135 + x105:x653 +
                  x604:x664 + x307:x427 + x105:x307 + x096:x457 + x685:x704 + x127:x693 +
                  x662:x749 + x093:x631 + x073:x146 + x080:x696 + x265:x661 + x514:x696 +
                  x111:x416 + x105:x111 + x114:x253 + x092:x274 + x135:x416 + x127:x653 +
                  x589:x704 + x026:x661 + x105:x364) %>% 
  step_nzv(all_predictors(), unique_cut = 1) %>% # remove predictors with near zero var 
  step_normalize(all_predictors()) # normalize again after PCA and transformations

# prep and bake
#recipe_pca %>% 
#  prep() %>% 
#  bake(new_data = NULL) %>% 
#  head(15) 
  
# save needed objects
save(best_pred, best_pred50, miss_combo, miss_combo50, skew_pred, 
     skew_pred50, recipe_pca, recipe_50, recipe_int, reg_fold,
     file = "results/modeling_objs.rda")