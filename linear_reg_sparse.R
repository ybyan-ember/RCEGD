library(MASS)
library(CVXR)

####parameter dimension####
d = 1000
####sample size####
n = 300
####sparsity####
s = 5
####true parameter####
theta_0 = rep(0,d)
a = sample(1:d, s, replace = FALSE)
for (i in 1:s){
  theta_0[a[i]] = (rbinom(1, 1, 0.5)-0.5)*2
}

####covariance matrix of covariate####
C = array(0, dim = c(d,d))
rho = 1/exp(1)
for (i in 1:d){
  for (j in 1:d){
    C[i,j] = rho^abs(i-j)
  }
} 
lambda_max = max(eigen(C)$val)

#################
####generate samples####
####inlier####
n1 = 270
X1 = mvrnorm(n1, mu=rep(0, d), Sigma=C)
s1 = 0.1
e = rnorm(n1, 0, s1)
y1 = X1 %*% theta_0 +e

####outlier####
n2 = 30
X2 = array(0, dim = c(n2,d))
for (i in 1:n2){
  for (j in 1:d){
    X2[i,j] = (rbinom(1, 1, 0.5)-0.5)*2
  }
}
y2 = -X2 %*% theta_0

####shuffle####
X_all = rbind(X1,X2)
y_all = rbind(y1,y2)
A = cbind(X_all,y_all) 
A_split = A[sample(1:n),]
#################

####Hard Thresholding####
hardthresh = function(s=5, x=1:10){
  d = length(x)
  a = order(abs(x), decreasing=TRUE)[1:s]
  u = rep(0,d)
  u[a] = x[a]
  return(u)
}

####robust-and-efficient hard thresholding for sparse linear regression####
regd<-function(T=20, x, theta0=rep(0,d), k=20, p=2, sp=10){
  #sp is the preset sparse number
  n = nrow(x);
  D = ncol(x)-1;
  a = floor(n/k);
  G = array(0, dim = c(a,D)); 
  U = array(0, dim = c(k,D)); 
  V = array(0, dim = c(k,D)); 
  theta = theta0; 
  G_re = rep(0,D); 
  l_regd = rep(0,T);
  eta = 1/lambda_max; 
  for (i in 1:T){
    for (j in 1:k){
      w = x[(1+(j-1)*a):(j*a),1:D]
      y = x[(1+(j-1)*a):(j*a),D+1]
      for (l in 1:a){
        G[l,] = (w[l,] %*% theta - y[l]) * w[l,]/a
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
    theta = hardthresh(s=sp,x=theta)
    l_regd[i] = sqrt(sum((theta - theta_0)^2))
  }
  out = list(theta=theta,l_regd=l_regd)
  return(out)
}

###MOM-based hard thresholding for sparse linear regression####
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

rgd<-function(T = 20, x, theta0 = rep(0,d), b = 20, sp=10){
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
        G[l,] = (u[l,] %*% theta - v[l]) * u[l,]/a
      }
      for (r in 1:D){
        mu[j,r] = mean(G[,r])
      }
    }
    G_r = centerPointSet(t(mu))
    theta = theta - eta * G_r
    theta = hardthresh(s=sp,x=theta)
    l_rgd[i] = sqrt(sum((theta - theta_0)^2))
  }
  out = list(theta=theta,l_rgd=l_rgd)
  return(out)
}












