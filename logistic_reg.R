library(MASS)
library(CVXR)

####sigmoid function
sigmoid = function(t=1){
  a = 1/(1+exp(-t))
  return(a)
}

####sample size####
n=1000

####parameter dimension####
d = 10

####true parameter####
theta_0 = rep(1/sqrt(d),d)

####proportion of outliers####
proportion = 0.1


#################
####generate samples (for one iteration)####
####inlier####
n1 = n * (1-proportion)

C = array(0, dim = c(d,d))
rho = 0.5
for (i in 1:d){
  for (j in 1:d){
    C[i,j] = rho^abs(i-j)
  }
} 

X = mvrnorm(n=n1, mu=rep(0, d), Sigma=C)
y = (as.matrix(sign(X %*% theta_0)) + rep(1,n1))/2

####outlier####
n2 = n - n1
n21 = floor(n2/2)
n22 = n2 - n21

X1 = mvrnorm(n=n21, mu=rep(0, d), Sigma=(d^2)*diag(d))
y1 = (rep(1,n21)-as.matrix(sign(X1 %*% theta_0)))/2
X2 = mvrnorm(n=n22, mu=rep(0, d), Sigma=(d^3)*diag(d))
y2 = (rep(1,n22)-as.matrix(sign(X1 %*% theta_0)))/2

####shuffle####
X_all = rbind(X,X1,X2)
y_all = rbind(y,y1,y2)
A = cbind(X_all,y_all) 
A_split = A[sample(1:n),]
A_X = A_split[,1:d]
A_y = A_split[,d+1]

C_hat = t(A_split[,1:d]) %*% A_split[,1:d]/n  #sample covariance matrix
lambda_max = max(eigen(C_hat)$val)  #largest eigenvalue  

####robust-and-efficient gradient descent for logistic regression####
regd<-function(x, epsilon=0.0005, T=100, theta0=rep(0,d), k=50, p=2,eta=0.1){
  #k is the number of block
  n = nrow(x);
  D = ncol(x)-1;
  a = floor(n/k);
  G = array(0, dim = c(a,D)); 
  U = array(0, dim = c(k,D)); 
  V = array(0, dim = c(k,D)); 
  theta = theta0; 
  G_re = rep(0,D); #
  l_regd = rep(0,T);
  eta = 1/lambda_max
  delta = 1
  i =1 
  while (delta > epsilon){
    for (j in 1:k){
      w = x[(1+(j-1)*a):(j*a),1:D]
      y = x[(1+(j-1)*a):(j*a),D+1]
      for (l in 1:a){
        G[l,] = (sigmoid(w[l,] %*% theta) - y[l]) * w[l,]
      }
      for (r in 1:D){
        U[j,r] = mean(G[,r])/sd(G[,r])^p
        V[j,r] = 1/sd(G[,r])^p
      }
    }
    for (f in 1:D){
      G_re[f] = sum(U[,f])/sum(V[,f])
    }
    theta = theta - eta * G_re
    l_regd[i] = sqrt(sum((theta - theta_0)^2))
    if (i==1){
      delta = l_regd[i]
    }
    else{
      delta = abs(l_regd[i] - l_regd[i-1])
    }
    i = i + 1
    out = list(theta=theta,l_regd=l_regd)
    if (i>T) break
  }
  return(out)
}

####MOM-based gradient descent for logistic regression####
centerPointSet = function(A){
  d = dim(A)[1]
  n = dim(A)[2]
  x = Variable(d)
  obj = 0
  for (i in 1:n) {
    obj = obj + p_norm(A[,i]-x, 2)
  }
  prob = Problem(Minimize(obj))
  result = solve(prob)
  center = result$getValue(x)
}

rgd<-function(x, T = 20, epsilon=0.0005, theta0 = rep(0,d), b = 20){
  n = nrow(x);
  D = ncol(x)-1;
  a = floor(n/b);
  G = array(0, dim = c(a,D));
  u = array(0, dim = c(a,D));
  v = rep(0, a);
  theta = theta0; 
  l_rgd = rep(0,T)
  G_r = rep(0,D); 
  mu = array(0, dim = c(b,D));
  eta = 1/lambda_max;
  delta = 1
  i =1 
  while (delta > epsilon){
    for (j in 1:b){
      u = x[(1+(j-1)*a):(j*a),1:D]
      v = x[(1+(j-1)*a):(j*a),D+1]
      for (l in 1:a){
        G[l,] = (sigmoid(u[l,] %*% theta) - v[l]) * u[l,]
      }
      for (r in 1:D){
        mu[j,r] = mean(G[,r])
      }
    }
    G_r = centerPointSet(t(mu))
    theta = theta - eta * G_r
    l_rgd[i] = sqrt(sum((theta - theta_0)^2))
    if (i==1){
      delta = l_rgd[i]
    }
    else{
      delta = abs(l_rgd[i] - l_rgd[i-1])
    }
    i = i + 1
    out = list(theta=theta,l_rgd=l_rgd)
    if (i>T) break
  }
  out = list(theta=theta,l_rgd=l_rgd)
  return(out)
}










