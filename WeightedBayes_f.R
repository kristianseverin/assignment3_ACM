WeightedBayes_f <- function(bias, Source1, Source2, w1, w2){
  
  outcome <- inv_logit_scaled(bias + w1*logit_scaled(Source1) + w2*logit_scaled(Source2))
  
  return(outcome)
  
}