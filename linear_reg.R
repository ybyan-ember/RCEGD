
# 更改

library(MASS)
library(CVXR)
####parameter dimension####
d = 10

####true parameter####
theta_0 = rep(1, d)

####covariance matrix of covariate####
C = array(0, dim = c(d,d))
rho = 0.5
for (i in 1:d){
  for (j in 1:d){
    C[i,j] = rho^abs(i-j)
  }
} 
lambda_max = max(eigen(C)$val)

####inlier####
generate_sample = function(n=500,s=1,rho = 0.5){
  C = array(0, dim = c(d,d))
  for (i in 1:d){
    for (j in 1:d){
      C[i,j] = rho^abs(i-j)
    }
  } 
  e = rnorm(n, 0, s)
  X = mvrnorm(n=n, mu=rep(0, d), Sigma=C)
  y = X%*%theta_0 +e
  A = cbind(X,y)
  return(A)
}

####outlier####
generate_outlier = function(n=500, mu=rep(0,d), r_1=100){
  X = mvrnorm(n=n, mu=mu, Sigma=r_1*diag(d));
  y = -X%*%theta_0
  A = cbind(X,y)
  return(A)
}

####outlier_2####
generate_outlier_2 = function(n=500, mu=rep(0,d), r_1=100){
  X = mvrnorm(n=n, mu=mu, Sigma=r_1*diag(d));
  y = rep(0,n)
  A = cbind(X,y)
  return(A)
}

####generate samples####
A_1 = generate_sample(n=900,s=0.1)
A_2 = generate_outlier(n=50,r_1=100)
A_3 = generate_outlier_2(n=50,r_1=100)
A = rbind(A_1,A_2,A_3)
A_split = A[sample(1:1000),]


####robust-and-efficient gradient descent for linear regression####
regd<-function(T=20, x, theta0=rep(0,d), k=50, p=2){
  #k is the number of block
  n = nrow(x);
  D = ncol(x)-1;
  a = floor(n/k);
  G = array(0, dim = c(a,D)); 
  U = array(0, dim = c(k,D)); 
  V = array(0, dim = c(k,D)); 
  theta = theta0; 
  G_re = rep(0,D); 
  l_regd = rep(0,T);
  eta = 1/lambda_max 
  for (i in 1:T){
    for (j in 1:k){
      w = x[(1+(j-1)*a):(j*a),1:D]
      y = x[(1+(j-1)*a):(j*a),D+1]
      for (l in 1:a){
        G[l,] = (w[l,] %*% theta - y[l]) * w[l,]
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
  }
  out = list(theta=theta,l_regd=l_regd)
  return(out)
}


####MOM-based gradient descent for linear regression####
####solve geometric median####
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

rgd<-function(T = 20, x, theta0 = rep(0,d), b = 20){
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
  for (i in 1:T){
    for (j in 1:b){
      u = x[(1+(j-1)*a):(j*a),1:D]
      v = x[(1+(j-1)*a):(j*a),D+1]
      for (l in 1:a){
        G[l,] = (u[l,] %*% theta - v[l]) * u[l,]
      }
      for (r in 1:D){
        mu[j,r] = mean(G[,r])
      }
    }
    G_r = centerPointSet(t(mu))
    theta = theta - eta * G_r
    l_rgd[i] = sqrt(sum((theta - theta_0)^2))
    print(l_rgd[i])
  }
  out = list(theta=theta,l_rgd=l_rgd)
  return(out)
  #return(theta)
}








