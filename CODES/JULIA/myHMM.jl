""" H I D D E N   M A R K O V   M O D E L S
    The following functions are taken from the source code of HMM R package
"""
workspace()

using StatsBase
using NamedArrays

function initHMM(states, symbols, startprobs = Union{}, transprobs = Union{}, emissprobs = Union{})
  """ HMM INITIALIZATION FUNCTION

  states - a column vector of either a string or numerics
  symbols - a column vector of either a string or numerics

  """
  n_states = (states |> size)[1]
  n_symbols = (symbols |> size)[1]

  S = NamedArray(repmat([1 / n_states], n_states))
  T = NamedArray(.5 * eye(n_states) + repmat([0.5 / n_states], n_states, n_states))
  E = NamedArray(repmat([1 / n_symbols], n_states, n_symbols))

  if startprobs != Union{}
    S = NamedArray(startprobs)
  end
  if transprobs != Union{}
    T = NamedArray(transprobs)
  end
  if emissprobs != Union{}
    E = NamedArray(emissprobs)
  end

  setnames!(S, states, 1)
  setnames!(T, states, 1)
  setnames!(T, states, 2)
  setnames!(E, states, 1)
  setnames!(E, symbols, 2)

  setdimnames!(S, "States", 1)
  setdimnames!(T, "from", 1)
  setdimnames!(T, "to", 2)
  setdimnames!(E, "States", 1)
  setdimnames!(E, "Symbols", 2)

  return Dict(
    :States => states,
    :Symbols => symbols,
    :Start_Probs => S,
    :Trans_Probs => T,
    :Emiss_Probs => E
    )
end

function simHMM(hmm, len)
  """ FUNCTION FOR SIMULATING HMM SEQUENCE

  hmm - object returned by initHMM
  len - length of the HMM sequence
  """

  states   = Array(AbstractString, 0)
  emission = Array(AbstractString, 0)
  append!(states, wsample(hmm[:States], vec(hmm[:Start_Probs]), 1))

  j = 1
  for i in 2:len
    append!(states, wsample(hmm[:States], vec(hmm[:Trans_Probs][states[i - 1], :]), 1))
    append!(emission, wsample(hmm[:Symbols], vec(hmm[:Emiss_Probs][states[j], :]), 1))
    j += 1
  end

  return Dict(
    :States => states,
    :Emission => emission
  )
end

function forward(hmm, obs)
  """ FUNCTION FOR COMPUTING THE FORWARD PROBABILITY

  hmm - object hmm
  obs - the observations given
  """
  n_obs = (obs |> size)[1]
  n_states = (hmm[:States] |> size)[1]
  α = NamedArray(zeros((n_states, n_obs)))
  setnames!(α, hmm[:States], 1)

  setdimnames!(α, "states", 1)
  setdimnames!(α, "index", 2)

  # Initialize
  for state in hmm[:States]
    α[state, 1] = log(hmm[:Start_Probs][state] * hmm[:Emiss_Probs][state, obs[1]])
  end

  # Iteration
  for k in 2:n_obs
    for state in hmm[:States]
      logsum = -Inf
      for prvstate in hmm[:States]
        temp = α[prvstate, k - 1] + log(hmm[:Trans_Probs][prvstate, state])
        if temp > -Inf
          logsum = temp + log(1 + exp(logsum - temp))
        end
      end
      α[state, k] = log(hmm[:Emiss_Probs][state, obs[k]]) + logsum
    end
  end

  return α
end

function backward(hmm, obs)
  """ FUNCTION FOR COMPUTING THE BACKWARD PROBABILITY

  hmm - object hmm
  obs - the observations given
  """
  n_obs = (obs |> size)[1]
  n_states = (hmm[:States] |> size)[1]
  β = NamedArray(zeros((n_states, n_obs)))
  setnames!(β, hmm[:States], 1)
  setdimnames!(β, "states", 1)
  setdimnames!(β, "index", 2)

  # Initialize
  for state in hmm[:States]
    β[state, n_obs] = log(1)
  end

  # Iteration
  for k in collect((n_obs - 1): -1 : 1)
    for state in hmm[:States]
      logsum = -Inf
      for nxtstate in hmm[:States]
        temp = β[nxtstate, k + 1] + log(hmm[:Trans_Probs][state, nxtstate] * hmm[:Emiss_Probs][nxtstate, obs[k + 1]])
        if temp > -Inf
          logsum = temp + log(1 + exp(logsum - temp))
        end
      end
      β[state, k] = logsum
    end
  end

  return β
end

function posterior(hmm, obs)
  α = forward(hmm, obs)
  β = backward(hmm, obs)

  prob_obs = α[1, (obs |> size)[1]]

  for i in 2:(hmm[:States] |> size)[1]
    j = α[i, (obs |> size)[1]]
    if j > -Inf
      prob_obs = j + log(1 + exp(prob_obs - j))
    end
  end

  post_prob = exp((α + β) - prob_obs)
  return post_prob
end

function baum_welch_recur(hmm, obs)
  """ FUNCTION FOR BAUM-WELCH RECURSION

  """
  n_obs = (obs |> size)[1]

  trans_mat = NamedArray(zeros(hmm[:Trans_Probs] |> size))
  emiss_mat = NamedArray(zeros(hmm[:Emiss_Probs] |> size))
  setnames!(trans_mat, allnames(hmm[:Trans_Probs])[1], 1)
  setnames!(trans_mat, allnames(hmm[:Trans_Probs])[2], 2)
  setnames!(emiss_mat, allnames(hmm[:Emiss_Probs])[1], 1)
  setnames!(emiss_mat, allnames(hmm[:Emiss_Probs])[2], 2)

  setdimnames!(trans_mat, dimnames(hmm[:Trans_Probs]))
  setdimnames!(emiss_mat, dimnames(hmm[:Emiss_Probs]))

  α = forward(hmm, obs)
  β = backward(hmm, obs)
  prob_obs = α[1, n_obs]

  for i in 2:(hmm[:States] |> size)[1]
    j = α[i, n_obs]

    if j > -Inf
      prob_obs = j + log(1 + exp(prob_obs - j))
    end
  end

  for x in hmm[:States]
    for y in hmm[:States]
      temp = α[x, 1] + log(hmm[:Trans_Probs][x, y]) +
        log(hmm[:Emiss_Probs][y, obs[1 + 1]]) + β[y, 1 + 1]

      for i in 2:((obs |> size)[1] - 1)
        j = α[x, i] + log(hmm[:Trans_Probs][x, y]) +
          log(hmm[:Emiss_Probs][y, obs[i + 1]]) + β[y, i + 1]

        if j > -Inf
          temp = j + log(1 + exp(temp - j))
        end
      end

      temp = exp(temp - prob_obs)
      trans_mat[x, y] = temp
    end
  end

  for x in hmm[:States]
    for s in hmm[:Symbols]
      temp = -Inf

      for i in 1:(obs |> size)[1]

        if s == obs[i]
          j = α[x, i] + β[x, i]
          if j > -Inf
            temp = j + log(1 + exp(temp - j))
          end
        end

      end

    temp = exp(temp - prob_obs)
    emiss_mat[x, s] = temp
    end
  end

  return Dict(
    :Trans_Mat => trans_mat,
    :Emiss_Mat => emiss_mat
    )

end

function baum_welch(hmm, obs, max_iter = 100, delta = 1e-9, pseudo_count = 0)
  dif = zeros(max_iter)

  for i in 1:max_iter
    # Expectation Step
    bw = baum_welch_recur(hmm, obs)
    T = bw[:Trans_Mat]
    E = bw[:Emiss_Mat]

    # Maximization Step (maximize log-likelihood for trans and emiss)
    T = NamedArray(T ./ sum(T, 2))
    E = NamedArray(E ./ sum(E, 2))

    d = ((hmm[:Trans_Probs] - T).^2 |> sum |> sqrt) + ((hmm[:Emiss_Probs] - E).^2 |> sum |> sqrt)
    dif[i] = d

    setnames!(T, allnames(hmm[:Trans_Probs])[1], 1)
    setnames!(T, allnames(hmm[:Trans_Probs])[2], 2)
    setnames!(E, allnames(hmm[:Emiss_Probs])[1], 1)
    setnames!(E, allnames(hmm[:Emiss_Probs])[2], 2)

    setdimnames!(T, dimnames(hmm[:Trans_Probs]))
    setdimnames!(E, dimnames(hmm[:Emiss_Probs]))

    hmm[:Trans_Probs] = T
    hmm[:Emiss_Probs] = E

    if d < delta
      break
    end
  end

  return Dict(:HMM => hmm, :Difference => dif)
end

function viterbi (hmm, obs)
  n_obs = (obs |> size)[1]
  n_states = (hmm[:States] |> size)[1]
  v = NamedArray(zeros(n_states, n_obs))

  setnames!(v, hmm[:States], 1)
  setdimnames!(v, "states", 1)
  setdimnames!(v, "index", 2)

  for state in hmm[:States]
    v[state, 1] = log(hmm[:Start_Probs][state] .* hmm[:Emiss_Probs][state, obs[1]])
  end

  for k in 2:n_obs
    for state in hmm[:States]
      maxi = NaN
      for prvstate in hmm[:States]
        temp = v[prvstate, k - 1] + log(hmm[:Trans_Probs][prvstate, state])
        maxi = maximum([maxi temp])
      end
      v[state, k] = log(hmm[:Emiss_Probs][state, obs[k]]) + maxi
    end
  end

  vit_path = repmat(["state"], n_obs)
  for state in hmm[:States]
    if maximum(v[:, n_obs]) == v[state, n_obs]
      vit_path[n_obs] = state
      break
    end
  end

  for k in collect((n_obs - 1) : -1 : 1)
    for state in hmm[:States]
      if maximum(v[:, k] + log(hmm[:Trans_Probs][:, vit_path[k + 1]])) == (v[state, k] + log(hmm[:Trans_Probs][state, vit_path[k + 1]]))
        vit_path[k] = state
        break
      end
    end
  end

  return vit_path
end

function viterbi_training (hmm, obs, max_iter = 100, delta = 1e-9, pseudo_count = 0)


end
hmm = initHMM(["AT-rich"; "GC-rich"], ["A"; "C"; "G"; "T"], Union{}, [[0.7 0.3]; [0.1  0.9]], [[0.39 0.10 0.10 0.41]; [0.10 0.41 0.39 0.10]])
obs = simHMM(hmm, 10)
est = baum_welch(hmm, obs[:Emission], 1000)






















est
viterbi(hmm, obs[:Emission])
viterbi(est[:HMM], obs[:Emission])

observations = ["T"; "H"; "H"; "T"; "H"; ""]
logForwardProbabilities = forward(hmm,observations)
exp(logForwardProbabilities)

baum_welch_recur(hmm, observations)
baum_welch(hmm, observations, 10)

viterbi(hmm, observations)

hmm[:Trans_Probs][:, ]


hmm = initHMM(["A"; "B"], ["L"; "R"], Union{}, [[.6 .4]; [.4 .6]], [[.6 .4]; [.4 .6]])

print(hmm)
# Sequence of observations
observations = ["L"; "L"; "R"; "R"]

obs = observations
n_obs = (obs |> size)[1]
n_states = (hmm[:States] |> size)[1]
v = NamedArray(zeros(n_states, n_obs))

setnames!(v, hmm[:States], 1)
setdimnames!(v, "states", 1)
setdimnames!(v, "index", 2)

for state in hmm[:States]
  v[state, 1] = log(hmm[:Start_Probs][state] .* hmm[:Emiss_Probs][state, obs[1]])
end
v
for k in 2:n_obs
  for state in hmm[:States]
    maxi = NaN
    for prvstate in hmm[:States]
      temp = v[prvstate, k - 1] + log(hmm[:Trans_Probs][prvstate, state])
      maxi = maximum([maxi temp])
    end
    v[state, k] = log(hmm[:Emiss_Probs][state, obs[k]]) + maxi
  end
end
v
vit_path = repmat(["state"], n_obs)
for state in hmm[:States]
  if maximum(v[:, n_obs]) == v[state, n_obs]
    vit_path[n_obs] = state
    break
  end
end
vit_path
for k in collect((n_obs - 1) : -1 : 1)
  for state in hmm[:States]
    if maximum(v[:, k] + log(hmm[:Trans_Probs][:, vit_path[k + 1]])) == (v[state, k] + log(hmm[:Trans_Probs][state, vit_path[k + 1]]))
      vit_path[k] = state
      break
    end
  end
end
vit_path
v
k = (n_obs - 1)
maximum(
v[:, k]
 +
 log(hmm[:Trans_Probs][:, vit_path[k + 1]])
)

log(hmm[:Trans_Probs][:, vit_path[1]])

vit_path[1]

vit_path = Array(AbstractString, 10)
vit_path = zeros((10, 1))
vit_path[1] = "a"



##############
hmm = initHMM(["A"; "B"], ["L"; "R"], Union{}, [[.6 .4]; [.4 .6]], [[.6 .4]; [.4 .6]])

print(hmm)
# Sequence of observations
observations = ["L"; "L"; "R"; "R"]

# Calculate Viterbi path
baum_welch(hmm, observations)
vit = viterbi(hmm, observations)
print(viterbi)
