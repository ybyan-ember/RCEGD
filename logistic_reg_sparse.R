library(MASS)
library(CVXR)

####sigmoid function
sigmoid = function(t=1){
  a = 1/(1+exp(-t))
  return(a)
}

###generate samples from sparse LDA problem
####parameter dimension####
d = 1000
####sample size####
n = 300
####sparsity####
s = 5   
####percentage of outliers####
epsilon = 0.1

#####true parameter####
theta_0 = rep(0,d)
a = sample(1:d, s, replace = FALSE)
for (i in 1:s){
  theta_0[a[i]] = (rbinom(1, 1, 0.5)-0.5)*2/sqrt(s)
}

####inlier####
n1 = n * (1-epsilon)
X1 = mvrnorm(n1, mu=rep(0, d), Sigma=diag(d))
y1 = (as.matrix(sign(X1 %*% theta_0)) + rep(1,n1))/2

lambda_max = max(eigen(diag(d))$val)

####outlier####
n2 = n - n1
X2 = array(0, dim = c(n2,d))
y2 = rep(0,n2)
for (i in 1:n2){
  for (j in 1:d){
    X2[i,j] = (rbinom(1, 1, 0.5)-0.5) 
  }
  y2[i] = (rep(1,n2)-sign(X2[i,] %*% theta_0))/2
}
y2 = as.vector(y2)


####shuffle####
A1 = cbind(X1,y1)
A2 = cbind(X2,y2)
A_split = rbind(A1,A2) 
A_split = A_split[sample(1:n),]

####Hard Thresholding####
hardthresh = function(s=5, x=1:10){
  d = length(x)
  a = order(abs(x), decreasing=TRUE)[1:s]
  u = rep(0,d)
  u[a] = x[a]
  return(u)
}

####robust-and-efficient hard thresholding for sparse logistic regression####
regd<-function(T=20, x, theta0=rep(0,d), k=20, p=2, sp=10){
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
        G[l,] = (sigmoid(w[l,] %*% theta) - y[l]) * w[l,]/a
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

###MOM-based hard thresholding for sparse logistic regression####
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
        G[l,] = (sigmoid(u[l,] %*% theta) - v[l]) * u[l,]/a
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














