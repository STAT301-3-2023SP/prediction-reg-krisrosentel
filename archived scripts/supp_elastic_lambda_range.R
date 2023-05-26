# Load package(s)
library(pacman)
p_load(tidymodels, tidyverse, glmnet, MASS)

# handle common conflicts
tidymodels_prefer()

# load data
reg_train <- read_csv("data/train.csv") 

# load saved objects from setup
load("results/modeling_objs.rda")

# set up interaction recipe
recipe_int_pre <- recipe_50 %>% 
  step_interact(~all_predictors()^2)

# prep and bake
clean_int_pre <- recipe_int_pre %>% 
  prep() %>% 
  bake(new_data = NULL) 

## work out best possible interactions for interaction recipe 
y <- clean_int_pre$y

x <- data.matrix(clean_int_pre[, c(clean_int_pre %>% 
                            select(-y) %>% 
                            colnames())])

# run cross validation to find the optimal value of lambda for lasso
set.seed(43)
elastic <- cv.glmnet(x, y, family = MASS::negative.binomial(3), alpha = 1, nfolds = 5)
plot(elastic)
elastic$lambda.min

# fit Lasso
lasso <- glmnet(x, y, alpha = 1, family = MASS::negative.binomial(3), lambda = .02588112)

# Look at best predictors
lasso %>% 
  tidy() %>% 
  mutate(abs_coef = abs(estimate)) %>% 
  arrange(desc(abs_coef)) %>% 
  print(n = 74)

# set up interaction recipe - narrowed
recipe_int # recipe with best interactions, moved building to EDA/setup

# prep and bake
clean_int <- recipe_int %>% 
  prep() %>% 
  bake(new_data = NULL) 

## work out possible range for elastic net for int recipe 
y <- clean_int$y

x <- data.matrix(clean_int[, c(clean_int %>% 
                                 select(-y) %>% 
                                 colnames())])
# run cross validation to find the optimal value of lambda for elastic net
set.seed(6)
elastic <- cv.glmnet(x, y, alpha = 0, nfolds = 5)
plot(elastic)
elastic$lambda.min     

# lm, lasso- .01880096, ridge - .4550072, mix - 0.04126805
# poisson, lasso- .03957421, ridge- .4550072, mix - .07914842  
# negative binomial, lasso- 0.002248812, ridge- 0.1145576, mix - 0.004497623  

## work out possible range for elastic net for pca recipe 
# prep and bake
clean_pca <- recipe_pca %>% 
  prep() %>% 
  bake(new_data = NULL) 

y <- clean_pca$y

x <- data.matrix(clean_pca[, c(clean_pca %>% 
                                 select(-y) %>% 
                                 colnames())])

# run cross validation to find the optimal value of lambda for elastic net
set.seed(12)
elastic <- cv.glmnet(x, y, alpha = 0, nfolds = 5)
plot(elastic)
elastic$lambda.min

# lm, lasso- .01566312, ridge - .4565883, mix - 0.02854331  
# poisson, lasso- 0.09173923, ridge- .5499622, mix - .1671787
# negative binomial, lasso- 0.001289993, ridge- .1148371, mix - 0.001225702   