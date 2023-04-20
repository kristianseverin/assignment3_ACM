# simple bayes multiple sources
SimpleBayes_MultiSource_f <- function(bias, sources) {
  
  outcome <- inv_logit_scaled(bias + sum(logit_scaled(sources)))
  
  return(outcome)
}