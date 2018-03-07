using Distributions
using Gadfly
using DataFrames
Gadfly.push_theme(:dark)

srand(12345);

x = rand(Normal(15, 2), 100);
beta = [3.4; .75];
err = rand(Normal(), 100);
X = [ones(length(x)) x];
y = X * beta + err;

xy_df = DataFrame(X = x, Y = y);
plot(xy_df, x = :X, y = :Y, Geom.point)
plot(xy_df, x = :Y, Geom.histogram(bincount = 10))


function v(S, vmax, k)
  (vmax * S) ./ (k + S)
end


x = collect(1:100);
y = v(x, maximum(x), 10) + randn(length(x));
z = v(x, maximum(x), .2 * maximum(y)) + rand(Normal(0, 3), length(x));

xy_df = DataFrame(X = x, Y = z);
plot(xy_df, x = :X, y = :Y, Geom.point)
