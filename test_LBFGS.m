function [] = test_LBFGS()
test_1();
test_2();
test_3();
test_4();
test_5();
test_6();
test_7();
test_8();
end

function test_1()
n = 100;
x0 = 5.0*ones(n,1);
l = -10.0*ones(n,1);
u = 10.0*ones(n,1);
x = LBFGSB(@quadratic,x0,l,u,[]);
x_exact = zeros(n,1);
if ( max(abs(x - x_exact)) < 1.0e-5 )
  fprintf('test 1: passed :)\n\n');
else
  fprintf('test 1: failed :(\n\n');
end
end

function test_2()
n = 100;
x0 = 5.0*ones(n,1);
l = ones(n,1);
u = 10.0*ones(n,1);
x = LBFGSB(@quadratic,x0,l,u,[]);
x_exact = ones(n,1);
if ( max(abs(x - x_exact)) < 1.0e-5 )
  fprintf('test 2: passed :)\n\n');
else
  fprintf('test 2: failed :(\n\n');
end
end

function test_3()
n = 100;
x0 = -20.0*ones(n,1);
l = -10.0*ones(n,1);
u = 10.0*ones(n,1);
x = LBFGSB(@quadratic,x0,l,u,[]);
x_exact = zeros(n,1);
if ( max(abs(x - x_exact)) < 1.0e-5 )
  fprintf('test 3: passed :)\n\n');
else
  fprintf('test 3: failed :(\n\n');
end
end

function test_4()
n = 100;
x0 = 1 + 9.0*rand(n,1);
l = ones(n,1);
u = 10.0*ones(n,1);
x = LBFGSB(@quadratic,x0,l,u,[]);
x_exact = ones(n,1);
if ( max(abs(x-x_exact)) < 1.0e-5 )
  fprintf('test 4: passed :)\n\n');
else
  fprintf('test 4: failed :(\n\n');
end
end

function test_5()
n = 100;
x0 = -2.0*ones(n,1);
l = 5.0*ones(n,1);
u = 10.0*ones(n,1);
x = LBFGSB(@neg_quadratic,x0,l,u,[]);
x_exact = 10.0*ones(n,1);
if ( max(abs(x - x_exact)) < 1.0e-5 )
  fprintf('test 5: passed :)\n\n');
else
  fprintf('test 5: failed :(\n\n');
end
end

function test_6()
n = 100;
x0 = 20.0*ones(n,1);
l = 5.0*ones(n,1);
u = 10.0*ones(n,1);
x = LBFGSB(@neg_quadratic,x0,l,u,[]);
x_exact = 10.0*ones(n,1);
if ( max(abs(x - x_exact)) < 1.0e-5 )
  fprintf('test 6: passed :)\n\n');
else
  fprintf('test 6: failed :(\n\n');
end
end

function test_7()
x0 = [8; 8];
l = [-inf; -inf];
u = [inf; inf];
opts = struct('max_iters',100);
x = LBFGSB(@rosenbrock,x0,l,u,opts);
x_exact = [1;1];
if ( max(abs(x - x_exact)) < 1.0e-5 )
  fprintf('test 7: passed :)\n\n');
else
  fprintf('test 7: failed :(\n\n');
end
end

function test_8()
x0 = [8; 8];
l = [2; 2];
u = [inf; inf];
opts = struct('max_iters',100);
x = LBFGSB(@rosenbrock,x0,l,u,opts);
x_exact = [2;4];
if ( max(abs(x - x_exact)) < 1.0e-5 )
  fprintf('test 8: passed :)\n\n');
else
  fprintf('test 8: failed :(\n\n');
end
end

function [f,g] = quadratic(x)
f = transpose(x)*x;
if (nargout > 1)
  g = 2.0*x;
end
end

function [f,g] = neg_quadratic(x)
f = -transpose(x)*x;
if (nargout > 1)
  g = -2.0*x;
end
end

function [f,g] = rosenbrock(x)
f = (1.0-x(1))*(1.0-x(1)) + 100.0*(x(2)-x(1)*x(1))*(x(2)-x(1)*x(1));
if (nargout > 1)
  g = zeros(2,1);
  g(1) = 2.0*(200.0*x(1)*x(1)*x(1) - 200.0*x(1)*x(2) + x(1) - 1.0);
  g(2) = 200.0*(x(2)-x(1)*x(1));
end
end