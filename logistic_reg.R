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

####percentage of outliers####
epsilon = 0.1


#################
####generate samples####
####inlier####
n1 = n * (1-epsilon)

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

X1 = mvrnorm(n=n2, mu=rep(0, d), Sigma=(d^3)*diag(d))
y1 = (rep(1,n2)-as.matrix(sign(X1 %*% theta_0)))/2

####shuffle####
X_all = rbind(X,X1)
y_all = rbind(y,y1)
A = cbind(X_all,y_all) 
A_split = A[sample(1:n),]
A_X = A_split[,1:d]
A_y = A_split[,d+1]
#################


####learning rate####
R = array(0, dim = c(n,n))
for (i in 1:n) {
  R[i,i] = sigmoid(X_all[i,] %*% theta_0) * (1-sigmoid(X_all[i,] %*% theta_0)) 
}
S = t(X_all) %*% R %*% X_all

lambda_max = max(eigen(S)$val)

####robust-and-efficient gradient descent for logistic regression####
regd<-function(T=20, x, theta0=rep(0,d), k=50, p=2,eta=0.1){
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
  for (i in 1:T){
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
  }
  out = list(theta=theta,l_regd=l_regd)
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

rgd<-function(T = 20, x, theta0 = rep(0,d), b = 20,eta=0.1){
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
        G[l,] = (sigmoid(u[l,] %*% theta) - v[l]) * u[l,]
      }
      for (r in 1:D){
        mu[j,r] = mean(G[,r])
      }
    }
    G_r = centerPointSet(t(mu))
    theta = theta - eta * G_r
    l_rgd[i] = sqrt(sum((theta - theta_0)^2))
  }
  out = list(theta=theta,l_rgd=l_rgd)
  return(out)
}










