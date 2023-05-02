# simple bayes function
SimpleBayes_f <- function(bias, Source1, Source2){
  
  outcome <- inv_logit_scaled(bias + logit_scaled(Source1) + logit_scaled(Source2))
  
  return(outcome)
  
}