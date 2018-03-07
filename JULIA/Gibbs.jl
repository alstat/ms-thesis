"""
GIBBS SAMPLING

This work is still on going.

(c) 2017 Al-Ahmadgaid B. Asaad
"""

immutable Gibbs
  cond_dist::Dict{String, Function}
  init_est ::Array{Float64}
end

function mcmc(parameters::Gibbs;
  set_seed::Int64 = 123,
  r::Int64 = 1000)

  cdist1, cdist2 = parameters.cond_dist["d1"], parameters.cond_dist["d2"]

  chain = zeros(r, 2)
  chain[1, 1] = cdist1(parameters.init_est[1])
  chain[1, 2] = cdist1(parameters.init_est[2])

  for i in 2:r
    chain[i, 1] = cdist1(chain[i, 2])
    chain[i, 2] = cdist2(chain[i, 1])
  end

  return chain

end
