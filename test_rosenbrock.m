function [] = test_rosenbrock()

% solve the unconstrained rosenbrock problem
x0 = [-1.2;1];
l = [-inf;-inf];
u = [inf;inf];
opts = struct('display',true,'xhistory',true,'max_iters',100);
func = @(x) rosenbrock(x(1),x(2));
[x,xhistory] = LBFGSB(func,x0,l,u,opts);
xhist_unc_x = xhistory(1,:);
xhist_unc_y = xhistory(2,:);

% unconstrained countour plot / search history
X = linspace(-1.5,1.5);
Y = linspace(-1,3);
[XX,YY] = meshgrid(X,Y);
FF = rosenbrock(XX,YY);
levels = 10:20:300;
figure
contour(X,Y,FF,levels,'linewidth',1.5)
colorbar
axis([-1.5 1.5 -1 3])
axis square
hold on
plot(x0(1),x0(2),'-rs','linewidth',3,'MarkerSize',10)
plot(xhist_unc_x,xhist_unc_y,'-x','linewidth',3,'MarkerSize',10)
plot(x(1),x(2),'-gs','linewidth',3,'MarkerSize',10)
hold on;
legend('countours f(x)', ...
  'initial x0', ...
  'search path', ...
  'final x');
set(gca,'fontsize',22)

% solve the constrained rosenbrock problem
x0 = [-1.2;1.0];
l = [-0.5;-0.5];
u = [0.5;0.5];
opts = struct('display',true,'xhistory',true,'max_iters',100);
func = @(x) rosenbrock(x(1),x(2));
[x,xhistory] = LBFGSB(func,x0,l,u,opts);
xhist_unc_x = xhistory(1,:);
xhist_unc_y = xhistory(2,:);

X = linspace(-1.5,1.5);
Y = linspace(-1,3);
[XX,YY] = meshgrid(X,Y);
FF = rosenbrock(XX,YY);
levels = 10:20:300;
figure
contour(X,Y,FF,levels,'linewidth',1.5)
colorbar
axis([-1.5 1.5 -1 3])
axis square
hold on
plot(x0(1),x0(2),'-rs','linewidth',3,'MarkerSize',10)
plot(xhist_unc_x,xhist_unc_y,'-x','linewidth',3,'MarkerSize',10)
plot(x(1),x(2),'-gs','linewidth',3,'MarkerSize',10)
x1=-0.5; x2=0.5; y1=-0.5; y2=0.5;
x = [x1, x2, x2, x1, x1];
y = [y1, y1, y2, y2, y1];
plot(x, y, '-o', 'LineWidth', 1,'MarkerSize',10);
hold on;
legend('countours f(x)', ...
  'initial x0', ...
  'search path', ...
  'final x', ...
  'feasible space')
set(gca,'fontsize',22)

% compare to fmincon
x0 = [-1.2;1.0];
l = [-0.5;-0.5];
u = [0.5;0.5];
opts = struct('display',false,'xhistory',false,'max_iters',100);
func = @(x) rosenbrock(x(1),x(2));
tic
LBFGSB(func,x0,l,u,opts);
toc
tic
fmincon(func,x0,[],[],[],[],l,u);
toc

end

function [f,g] = rosenbrock(x,y)
f = (1.0-x).^2 + 100.0*(y-x.*x).^2;
if (nargout > 1)
  g = zeros(2,1);
  g(1) = 2.0*(200.0.*x.*x.*x - 200.0.*x.*y + x - 1.0);
  g(2) = 200.0*(y-x.*x);
end
end