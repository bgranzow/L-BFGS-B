function [x,xhist] = LBFGSB(func,x0,l,u,options)
% function [x,xhist] = LBFGSB(func,x0,l,u,options)
% Perform bound-constrained optimization with L-BFGS-B.
% INPUTS:
%  x0: [n,1] initial design vector.
%  l: [n,1] lower bound constraint vector.
%  u: [n,1] upper bound constraint vector.
%  options: matlab struct with the optional components:
%   'm': the maximum number of stored L-BFGS iteration pairs.
%   'tol': the convergence tolerance for the projected gradient.
%   'display': true/false - should iterations be displayed?
%   'xhist': true/false - should the entire search history be stored?

% validate the inputs
[x0] = validate_inputs(func,x0,l,u);

% set options
[m,tol,max_iters,display,xhistory] = set_options(options);

% initialize BFGS variables
n = length(x0);
Y = zeros(n,0);
S = zeros(n,0);
W = zeros(n,1);
M = zeros(1,1);
theta = 1;

% initialize objective variables
x = x0;
[f,g] = feval(func, x);

% initialize quasi-Newton iterations
k = 0;

% print out some useful information, if specified
if (display)
  fprintf(' iter        f(x)          optimality\n')
  fprintf('-------------------------------------\n')
  opt = get_optimality(x,g,l,u);
  fprintf('%3d %16.8f %16.8f\n',k,f,opt);
end

% save the xhistory, if specified
xhist = [];
if (xhistory)
  xhist = [xhist x0];
end

% perform quasi-Newton iterations
while ( (get_optimality(x,g,l,u) > tol) && (k < max_iters) )
  
  % update search information
  x_old = x;
  g_old = g;
  
  % compute the new search direction
  [xc, c] = get_cauchy_point(x,g,l,u,theta,W,M);
  [xbar, line_search_flag] = subspace_min(x,g,l,u,xc,c,theta,W,M);

  alpha = 1.0;
  if (line_search_flag)
    [alpha] = strong_wolfe(func,x,f,g,xbar-x);
  end
  x = x + alpha * (xbar - x);
  
  % update the LBFGS data structures
  [f,g] = feval(func, x);
  y = g - g_old;
  s = x - x_old;
  curv = (transpose(s)*y); % NOTE: keep sign of curvature
  if (curv < eps)
    fprintf(' warning: negative curvature detected\n');
    fprintf('          skipping L-BFGS update\n');
    k = k+1;
    continue;
  end
  if (k < m)
    Y = [Y y];
    S = [S s];
  else
    Y(:,1:m-1) = Y(:,2:end);
    S(:,1:m-1) = S(:,2:end);
    Y(:,end) = y;
    S(:,end) = s;
  end
  theta = (transpose(y)*y)/(transpose(y)*s);
  W = [Y theta*S];
  A = transpose(S)*Y;
  L = tril(A,-1);
  D = -1*diag(diag(A));
  MM = [D transpose(L); L theta*transpose(S)*S];
  M = inv(MM);
  
  % update the iteration
  k = k+1;
  if (xhistory)
    xhist = [xhist x];
  end
  
  % print some useful information
  if (display)
    opt = get_optimality(x,g,l,u);
    fprintf('%3d %16.8f %16.8f\n',k,f,opt);
  end
  
end

if (k == max_iters)
  fprintf(' warning: maximum number of iterations reached\n')
end

if ( get_optimality(x,g,l,u) < tol )
  fprintf(' stopping because convergence tolerance met!\n')
end

end

function [x0] = validate_inputs(func,x0,l,u)
% function [x0] = validate_inputs(func,x0,l,u)
% Validate the inputs to the LBFGSB algorithm.
% INPUTS:
%  func: function handle to the objective, returns [f,g].
%  x0: [n,1] initial design vector.
%  l: [n,1] lower bound vector.
%  u: [n,1] upper bound vector.
% OUTPUTS:
%  none

% sanity check of inputs
if ( nargout(func) ~=2 && nargout(func) ~= -1 )
  error('input func return must be of form [f,g]');
end
if ( ~ iscolumn(x0) )
  error('input x0 must be a column vector')
end
if ( ~ iscolumn(l) )
  error('input l must be a column vector')
end
if ( ~ iscolumn(u) )
  error('input u must be a column vector')
end
if ( length(l) ~= length(x0) )
  error('input l must be of equal length to input x0');
end
if ( length(u) ~= length(x0) )
  error('input u must be of equal length to input xo');
end

% pull back x0 into the feasible design space if needed
modified_x0 = false;
for i=1:length(x0)
  if ( x0(i) < l(i) )
    x0(i) = l(i);
    modified_x0 = true;
  elseif ( x0(i) > u(i) )
    x0(i) = u(i);
    modified_x0 = true;
  end
end
if (modified_x0)
  fprintf(' note: initial guess x0 outside of bounds\n');
  fprintf('       projecting x0 back into the feasible space\n');
end

end

function [m,tol,max_iters,display,xhistory] = set_options(options)
% function [m,tol,max_iters] = set_options(options)
% Set optionally defined user input parameters.
% INPUTS:
%  options: a matlab struct ('m','tol','max_iters').
% OUTPUTS:
%  m: the limited memory storage size.
%  tol: the convergence tolerance criteria.
%  max_iters: the maximum number of quasi-Newton iterations.
%  display: true/false should iteration information be displayed?
%  xhistory: true/false should the entire search history be stored?
m = 10;
tol = 1.0e-5;
max_iters = 20;
display = true;
xhistory = false;
if ( isfield(options, 'm') )
  m = options.m;
end
if ( isfield(options, 'tol') )
  tol = options.tol;
end
if ( isfield(options, 'max_iters') )
  max_iters = options.max_iters;
end
if ( isfield(options, 'display') )
  display = options.display;
end
if ( isfield(options, 'xhistory') )
  xhistory = options.xhistory;
end

end

function [opt] = get_optimality(x,g,l,u)
% function [opt] = get_optimality(x,g,l,u)
% Get the inf-norm of the projected gradient.
% Equation (6.1), Page 17.
% INPUTS:
%  x: [n,1] design vector.
%  g: [n,1] objective function gradient.
%  l: [n,1] lower bound vector.
%  u: [n,1] upper bound vector.
% OUTPUTS:
%  opt: the inf-norm of the projected gradient.

projected_g = x-g;
for i=1:length(x)
  if (projected_g(i) < l(i))
    projected_g(i) = l(i);
  elseif (projected_g(i) > u(i))
    projected_g(i) = u(i);
  end
end
projected_g = projected_g - x;
opt = max(abs(projected_g));

end

function [t,d,F] = get_breakpoints(x,g,l,u)
% function [t,d,F] = get_breakpoints(x,g,l,u)
% Compute the breakpoint variables needed for the Cauchy point.
% Equations (4.1),(4.2), and F in Algorigthm CP: Initialize.
% INPUTS:
%  x: [n,1] design vector.
%  g: [n,1] objective gradient.
%  l: [n,1] lower bound vector.
%  u: [n,1] upper bound vector.
% OUTPUTS:
%  t: [n,1] breakpoint vector.
%  d: [n,1] search direction vector.
%  F: [n,1] the indices that sort t from low to high.

n = length(x);
t = zeros(n,1);
d = -g;
for i=1:n
  if ( g(i) < 0 )
    t(i) = ( x(i) - u(i) ) / g(i);
  elseif ( g(i) > 0 )
    t(i) = ( x(i) - l(i) ) / g(i);
  else
    t(i) = realmax;
  end
  if ( t(i) < eps )
    d(i) = 0.0;
  end
end
tuple = [t linspace(1,n,n)'];
tuple = sortrows(tuple);
F = tuple(:,2);

end

function [xc, c] = get_cauchy_point(x,g,l,u,theta,W,M)
% function [xc, c] = get_cauchy_point(x,g,l,u,theta,W,M)
% Compute the generalized Cauchy point.
% Algorithm CP, Pages 8-9.
% INPUTS:
%  x: [n,1] design vector.
%  g: [n,1] objective function gradient.
%  l: [n,1] lower bound vector.
%  u: [n,1] uppder bound vector.
%  theta: positive BFGS scaling.
%  W: [n,2m] BFGS matrix storage.
%  M: [2m,2m] BFGSB matrix storage.
% OUTPUTS:
%  xc - [n,1] the generalized Cauchy point.
%  c - [2m,1] initialization vector for subspace minimization.

% perform the initialization step
[tt,d,F] = get_breakpoints(x,g,l,u);
xc = x;
p = transpose(W) * d;
c = zeros(size(W,2),1);
fp = -transpose(d)*d;
fpp = -theta*fp - transpose(p)*M*p;
fpp0 = -theta*fp;
dt_min = -fp/fpp;
t_old = 0;
for j=1:length(x)
  i = j;
  if (F(i) > 0)
    break;
  end
end
b = F(i);
t = tt(b);
dt = t-t_old;

% examine the subsequent segments
while ( (dt_min > dt) && (i <= length(x)) )
  if ( d(b) > 0)
    xc(b) = u(b);
  elseif ( d(b) < 0)
    xc(b) = l(b);
  end
  zb = xc(b) - x(b);
  c = c + dt*p;
  gb = g(b);
  wbt = W(b,:);
  fp = fp + dt*fpp + gb*gb + theta*gb*zb - gb*wbt*(M*c);
  fpp = fpp - theta*gb*gb - 2.0*gb*wbt*(M*p) - gb*gb*wbt*(M*transpose(wbt));
  fpp = max( eps*fpp0, fpp );
  p = p + gb*transpose(wbt);
  d(b) = 0.0;
  dt_min = -fp/fpp;
  t_old = t;
  i = i+1;
  if (i <= length(x))
    b = F(i);
    t = tt(b);
    dt = t - t_old;
  end
end

% perform final updates
dt_min = max(dt_min, 0);
t_old = t_old + dt_min;
for j=i:length(xc)
  idx = F(j);
  xc(idx) = x(idx) + t_old*d(idx);
end
c = c + dt_min*p;

end

function [alpha_star] = find_alpha(l,u,xc,du,free_vars_idx)
% function [alpha_star] = find_alpha(l,u,xc,du,free_vars_idx)
% Equation (5.8), Page 8.
% INPUTS:
%  l: [n,1] lower bound constraint vector.
%  u: [n,1] upper bound constraint vector.
%  xc: [n,1] generalized Cauchy point.
%  du: [num_free_vars,1] solution of unconstrained minimization.
% OUTPUTS:
%  alpha_star: positive scaling parameter.

alpha_star = 1;
n = length(free_vars_idx);
for i=1:n
  idx = free_vars_idx(i);
  if (du(i) > 0)
    alpha_star = min(alpha_star, ( u(idx)-xc(idx) )/du(i) );
  else
    alpha_star = min(alpha_star, ( l(idx)-xc(idx) )/du(i) );
  end
end

end

function [xbar, line_search_flag] = subspace_min(x,g,l,u,xc,c,theta,W,M)
% function [xbar] = subspace_min(x,g,l,u,xc,c,theta,W,M)
% Subspace minimization for the quadratic model over free variables.
% Direct Primal Method, Page 12.
% INPUTS:
%  x: [n,1] design vector.
%  g: [n,1] objective gradient vector.
%  l: [n,1] lower bound vector.
%  u: [n,1] uppder bound vector.
%  xc: [n,1] generalized Cauchy point.
%  c: [2m,1] minimization initialization vector.
%  theta: positive LBFGS scaling parameter.
%  W: [n,2m] LBFGS matrix storage.
%  M: [2m,2m] LBFGS matrix storage.

% set the line search flag to true
line_search_flag = true;

% compute the free variables.
n = length(x);
free_vars_idx = [];
Z = [];
for i=1:length(xc)
  if ( ( xc(i) ~= u(i) ) && ( xc(i) ~= l(i) ) )
    free_vars_idx = [free_vars_idx i];
  end
end
num_free_vars = length(free_vars_idx);

if (num_free_vars == 0)
  xbar = xc;
  line_search_flag = false;
  return;
end

% compute W^T Z, the restriction of W to free variables.
WTZ=zeros(length(c),num_free_vars); % length(c)=2*m
for i=1:num_free_vars,
  WTZ(:,i) = W(free_vars_idx(i),:);
end

% compute the reduced gradient of mk restricted to free variables.
rr = g + theta*(xc-x) - W*(M*c);
r = zeros(num_free_vars, 1);
for i=1:num_free_vars
  r(i) = rr(free_vars_idx(i));
end

% form intermediate variables.
invtheta = 1.0/theta;
v = M*(WTZ*r);
N = invtheta*WTZ*transpose(WTZ);
N = eye(size(N)) - M*N;
v = N\v;
du = -invtheta*r - invtheta^2 * transpose(WTZ)*v;

% find alpha star
alpha_star = find_alpha(l,u,xc,du,free_vars_idx);

% compute the subspace minimization
d_star = alpha_star*du;
xbar = xc;
for i=1:num_free_vars
  idx = free_vars_idx(i);
  xbar(idx) = xbar(idx) + d_star(i);
end

end

function [alpha] = strong_wolfe(func,x0,f0,g0,p)
% function [alpha] = strong_wolfe(func,x0,f0,g0,p)
% Compute a line search to satisfy the strong Wolfe conditions.
% Algorithm 3.5. Page 60. "Numerical Optimization". Nocedal & Wright.
% INPUTS:
%  func: objective function handle.
%  x0: [n,1] initial design vector.
%  f0: initial function evaluation.
%  g0: [n,1] initial objective gradient vector.
%  p: [n,1] search direction vector.
% OUTPUTS:
% alpha: search length

% initialize variables
c1 = 1e-4;
c2 = 0.9;
alpha_max = 2.5;
alpha_im1 = 0;
alpha_i = 1;
f_im1 = f0;
dphi0 = transpose(g0)*p;
i = 0;
max_iters = 20;

% search for alpha that satisfies strong-Wolfe conditions
while true
  
  x = x0 + alpha_i*p;
  [f_i,g_i] = feval(func, x);
  if (f_i > f0 + c1*dphi0) || ( (i > 1) && (f_i >= f_im1) )
    alpha = alpha_zoom(func,x0,f0,g0,p,alpha_im1,alpha_i);
    break;
  end
  dphi = transpose(g_i)*p;
  if ( abs(dphi) <= -c2*dphi0 )
    alpha = alpha_i;
    break;
  end
  if ( dphi >= 0 )
    alpha = alpha_zoom(func,x0,f0,g0,p,alpha_i,alpha_im1);
    break;
  end
  
  % update
  alpha_im1 = alpha_i;
  f_im1 = f_i;
  alpha_i = alpha_i + 0.8*(alpha_max-alpha_i);
  
  if (i > max_iters)
    alpha = alpha_i;
    break;
  end
  
  i = i+1;
  
end

end

function [alpha] = alpha_zoom(func,x0,f0,g0,p,alpha_lo,alpha_hi)
% function [alpha] = alpha_zoom(func,x0,f0,g0,p,alpha_lo,alpha_hi)
% Algorithm 3.6, Page 61. "Numerical Optimization". Nocedal & Wright.
% INPUTS:
%  func: objective function handle.
%  x0: [n,1] initial design vector.
%  f0: initial objective value.
%  g0: [n,1] initial objective gradient vector.
%  p: [n,1] search direction vector.
%  alpha_lo: low water mark for alpha.
%  alpha_hi: high water mark for alpha.
% OUTPUTS:
%  alpha: zoomed in alpha.

% initialize variables
c1 = 1e-4;
c2 = 0.9;
i = 0;
max_iters = 20;
dphi0 = transpose(g0)*p;

while true
  alpha_i = 0.5*(alpha_lo + alpha_hi);
  alpha = alpha_i;
  x = x0 + alpha_i*p;
  [f_i,g_i] = feval(func,x);
  x_lo = x0 + alpha_lo*p;
  f_lo = feval(func, x_lo);
  if ( (f_i > f0 + c1*alpha_i*dphi0) || ( f_i >= f_lo) )
    alpha_hi = alpha_i;
  else
    dphi = transpose(g_i)*p;
    if ( ( abs(dphi) <= -c2*dphi0 ) )
      alpha = alpha_i;
      break;
    end
    if ( dphi * (alpha_hi-alpha_lo) >= 0 )
      alpha_hi = alpha_lo;
    end
    alpha_lo = alpha_i;
  end
  i = i+1;
  if (i > max_iters)
    alpha = alpha_i;
    break;
  end
end

end
