// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;  // number of trials
  array[N] int rating; // rating of trustworthiness for each 'N'
  array[N] real<lower=0, upper = 1> Source1;  // rating scaled 0.1 - 0.9
  array[N] real<lower=0, upper = 1> Source2;  // feedback scaled 0.1 - 0.9
  
}

// transformed data
transformed data{
  array[N] real l_Source1;
  array[N] real l_Source2;
  l_Source1 = logit(Source1);  // get it on an -inf - inf scale
  l_Source2 = logit(Source2);  // get it on an -inf - inf scale
}


// The parameters accepted by the model. Our model
// accepts one parameters 'bias' 
// i.e., do you have a prevaleence of rating faces either very or very little trustworthy
parameters {
  real bias;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  target +=  normal_lpdf(bias | 0, 1);
  target +=  normal_lpdf(rating | bias + to_vector(l_Source1) + to_vector(l_Source2) | 0, 3.5);
}

generated quantities{
  real bias_prior;
  array[N] real log_lik;
  
  bias_prior = normal_rng(0, 1);
  
  for (n in 1:N){  
    log_lik[n] = bernoulli_logit_lpmf(rating[n] | bias + l_Source1[n] +  l_Source2[n]);
  }
  
}
