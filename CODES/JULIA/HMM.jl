""" H I D D E N   M A R K O V   M O D E L S
    The following functions are taken from the source code of HMM R package
"""
workspace()

using StatsBase
using NamedArrays
using Distributions

function init_HMM(states, symbols = Union{}; startprobs = Union{}, transprobs = Union{}, emissprobs = Union{}, emiss_dist_type = "gaussian")
  """ HMM INITIALIZATION FUNCTION

  states - AbstractVector of either a string or numerics
  symbols - AbstractVector of either a string or numerics
  startprobs - AbstractVector
  transprobs - AbstractVector
  emissprobs - AbstractVector or Distribution Function
  """
  n_states = (states |> size)[1]

  if (symbols != Union{}) & (emissprobs == Union{})
    n_symbols = (symbols |> size)[1]
    emiss_dist_type = "multinomial"
  end

  init_start = NamedArray(repmat([1 / n_states], n_states))
  init_trans = NamedArray(.5 * eye(n_states) + repmat([0.5 / n_states], n_states, n_states))

  if emissprobs == Union{}
    if emiss_dist_type == "gaussian"
      init_emiss = Dict{AbstractString, Distributions.Distribution}()
      for i in states
        init_emiss[i] = Distributions.Normal()
      end
    elseif emiss_dist_type == "multinomial"
      init_emiss = NamedArray(repmat([1 / n_symbols], n_states, n_symbols))

      setnames!(init_emiss, states, 1)
      setnames!(init_emiss, symbols, 2)

      setdimnames!(init_emiss, "States", 1)
      setdimnames!(init_emiss, "Symbols", 2)
    else
      stop("emiss_dist_type values: gaussian or multinomial")
    end
  elseif (emissprobs != Union{}) & isa(emissprobs, Distributions.ContinuousDistribution) & !isa(emissprobs, AbstractArray)
    init_emiss = emissprobs
  elseif (emissprobs != Union{}) & !isa(emissprobs, Distributions.ContinuousDistribution) & isa(emissprobs, AbstractArray)
    init_emiss = NamedArray(emissprobs)

    setnames!(init_emiss, states, 1)
    setnames!(init_emiss, symbols, 2)

    setdimnames!(init_emiss, "States", 1)
    setdimnames!(init_emiss, "Symbols", 2)
  end

  if startprobs != Union{}
    init_start = NamedArray(startprobs)
  end
  if transprobs != Union{}
    init_trans = NamedArray(transprobs)
  end

  setnames!(init_start, states, 1)
  setnames!(init_trans, states, 1)
  setnames!(init_trans, states, 2)

  setdimnames!(init_start, "States", 1)
  setdimnames!(init_trans, "from", 1)
  setdimnames!(init_trans, "to", 2)

  return Dict(
    :States => states,
    :Symbols => symbols,
    :Start_Probs => init_start,
    :Trans_Probs => init_trans,
    :Emiss_Probs => init_emiss
    )
end

function sim_HMM(hmm, len)
  """ FUNCTION FOR SIMULATING HMM SEQUENCE

  hmm - object returned by initHMM
  len - length of the HMM sequence
  """
  emiss_dist = hmm[:Emiss_Probs]

  emissions = zeros(len)
  states = repmat(["state"], len)
  states[1] = wsample(hmm[:States], hmm[:Start_Probs], 1)[1]
  if isa(emiss_dist, Dict)
    emissions[1] = rand(emiss_dist[states[1]], 1)[1]
    for i in 2:len
      states[i] = wsample(hmm[:States], vec(hmm[:Trans_Probs][states[i - 1], :]), 1)[1]
      emissions[i] = rand(emiss_dist[states[i]], i)[1]
    end
  elseif isa(emiss_dist, AbstractArray)
    emissions[1] = wsample(hmm[:Symbols], vec(hmm[:Emiss_Probs][states[1], :]), 1)
    for i in 2:len
      states[i] = wsample(hmm[:States], vec(hmm[:Trans_Probs][states[i - 1], :]), 1)[1]
      emissions[i] = wsample(hmm[:Symbols], vec(hmm[:Emiss_Probs][states[i], :]), 1)
    end
  else
    error("Emiss_Probs must either a type of Dict containing the Distributions\n
      or AbstractArray containing the emission probabilities of the symbols")
  end

  return Dict(
    :States => states,
    :Emission => emissions
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

  emiss_dist = hmm[:Emiss_Probs]

  # Initialize
  for state in hmm[:States]
    if isa(emiss_dist, Dict)
      α[state, 1] = log(hmm[:Start_Probs][state] * pdf(emiss_dist[state], obs[1]))
    elseif isa(emiss_dist, AbstractArray)
      α[state, 1] = log(hmm[:Start_Probs][state] * hmm[:Emiss_Probs][state, obs[1]])
    else
      error("Emiss_Probs must either a type of Dict containing the Distributions\n
        or AbstractArray containing the emission probabilities of the symbols")
    end
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
      if isa(emiss_dist, Dict)
        α[state, k] = log(pdf(emiss_dist[state], obs[k])) + logsum
      elseif isa(emiss_dist, AbstractArray)
        α[state, k] = log(hmm[:Emiss_Probs][state, obs[k]]) + logsum
      end
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

  emiss_dist = hmm[:Emiss_Probs]

  # Iteration
  for k in collect((n_obs - 1): -1 : 1)
    for state in hmm[:States]
      logsum = -Inf
      for nxtstate in hmm[:States]
        if isa(emiss_dist, Dict)
          temp = β[nxtstate, k + 1] + log(hmm[:Trans_Probs][state, nxtstate] * pdf(emiss_dist[nxtstate], obs[k + 1]))
        elseif isa(emiss_dist, AbstractArray)
          temp = β[nxtstate, k + 1] + log(hmm[:Trans_Probs][state, nxtstate] * hmm[:Emiss_Probs][nxtstate, obs[k + 1]])
        else
          error("Emiss_Probs must either a type of Dict containing the Distributions\n
            or AbstractArray containing the emission probabilities of the symbols")
        end

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

function viterbi(hmm, obs)
  n_obs = (obs |> size)[1]
  n_states = (hmm[:States] |> size)[1]
  δ = NamedArray(zeros(n_states, n_obs))

  setnames!(δ, hmm[:States], 1)
  setdimnames!(δ, "states", 1)
  setdimnames!(δ, "index", 2)

  emiss_dist = hmm[:Emiss_Probs]
  for state in hmm[:States]
    if isa(emiss_dist, Dict)
      δ[state, 1] = log(hmm[:Start_Probs][state] * pdf(hmm[:Emiss_Probs][state], obs[1]))
    elseif isa(emiss_dist, AbstractArray)
      δ[state, 1] = log(hmm[:Start_Probs][state] * hmm[:Emiss_Probs][state, obs[1]])
    end
  end

  for k in 2:n_obs
    for state in hmm[:States]
      maxi = NaN
      for prvstate in hmm[:States]
        temp = δ[prvstate, k - 1] + log(hmm[:Trans_Probs][prvstate, state])
        maxi = maximum([maxi temp])
      end
      if isa(emiss_dist, Dict)
        δ[state, k] = log(pdf(hmm[:Emiss_Probs][state], obs[k])) + maxi
      elseif isa(emiss_dist, AbstractArray)
        δ[state, k] = log(hmm[:Emiss_Probs][state, obs[k]]) + maxi
      end
    end
  end

  ρ = repmat(["state"], n_obs)
  for state in hmm[:States]
    if maximum(δ[:, n_obs]) == δ[state, n_obs]
      ρ[n_obs] = state
      break
    end
  end

  for k in collect((n_obs - 1) : -1 : 1)
    for state in hmm[:States]
      if maximum(vec(δ[:, k]) + vec(log(hmm[:Trans_Probs][:, ρ[k + 1]]))) == (δ[state, k] + log(hmm[:Trans_Probs][state, ρ[k + 1]]))
        ρ[k] = state
        break
      end
    end
  end

  return ρ
end

function baum_welch_recur(hmm, obs)
  """ FUNCTION FOR BAUM-WELCH RECURSION
  hmm -
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

hmm = init_HMM(["C1", "C2"])
sim_hmm = sim_HMM(hmm, 100)
obs = sim_hmm[:Emission]
forward(hmm, sim_hmm[:Emission]) |> exp
backward(hmm, sim_hmm[:Emission]) |> exp
posterior(hmm, sim_hmm[:Emission])
viterbi(hmm, obs)

89 - 0


hmm = init_HMM(["AT-rich"; "GC-rich"], ["A"; "C"; "G"; "T"], startprobs = Union{}, transprobs = [[0.7 0.3]; [0.1  0.9]], emissprobs = [[0.39 0.10 0.10 0.41]; [0.10 0.41 0.39 0.10]])
obs = sim_HMM(hmm, 10)

est = baum_welch(hmm, obs[:Emission], 1000)

states = ["AT-rich"; "GC-rich"]
symbols = ["A"; "C"; "G"; "T"]
symbols = Union{}
startprobs = Union{}
transprobs = [[0.7 0.3]; [0.1  0.9]] #Union{}
emissprobs = [[0.39 0.10 0.10 0.41]; [0.10 0.41 0.39 0.10]]#Union{} #
emiss_dist_type = "gaussian"
len = 10
""" FUNCTION FOR SIMULATING HMM SEQUENCE

hmm - object returned by initHMM
len - length of the HMM sequence
"""
emiss_dist = hmm[:Emiss_Probs]

emissions = zeros(len)
states = repmat(["state"], len)
states[1] = wsample(hmm[:States], hmm[:Start_Probs], 1)[1]
if isa(emiss_dist, Dict)
  emissions[1] = rand(emiss_dist[states[1]], 1)[1]
  for i in 2:len
    states[i] = wsample(hmm[:States], vec(hmm[:Trans_Probs][states[i - 1], :]), 1)[1]
    emissions[i] = rand(emiss_dist[states[i]], i)[1]
  end
elseif isa(emiss_dist, AbstractArray)
  emissions[1] = wsample(hmm[:Symbols], vec(hmm[:Emiss_Probs][states[1], :]), 1)
  for i in 2:len
    states[i] = wsample(hmm[:States], vec(hmm[:Trans_Probs][states[i - 1], :]), 1)[1]
    emissions[i] = wsample(hmm[:Symbols], vec(hmm[:Emiss_Probs][states[i], :]), 1)
  end
else
  error("Emiss_Probs must either a type of Dict containing the Distributions\n
    or AbstractArray containing the emission probabilities of the symbols")
end

return Dict(
  :States => states,
  :Emission => emissions
)
































ρ
maximum(vec(δ[:, k]) + vec(log(hmm[:Trans_Probs][:, ρ[k + 1]])))
vec(δ[:, k]) + vec(log(hmm[:Trans_Probs][:, ρ[k + 1]]))
(δ[state, k] + log(hmm[:Trans_Probs][state, ρ[k + 1]]))
return ρ
δ[state, k]

n_obs = (obs |> size)[1]
n_states = (hmm[:States] |> size)[1]
v = NamedArray(zeros(n_states, n_obs))

setnames!(v, hmm[:States], 1)
setdimnames!(v, "states", 1)
setdimnames!(v, "index", 2)
v
state = hmm[:States][1]
obs[1]
log(hmm[:Start_Probs][state] .* hmm[:Emiss_Probs][state, obs[1]])
emiss_dist = hmm[:Emiss_Probs]
if isa(emiss_dist, Dict)
  for state in hmm[:States]
    v[state, 1] = log(hmm[:Start_Probs][state] * pdf(hmm[:Emiss_Probs][state], obs[1]))
  end

  for k in 2:n_obs
    for state in hmm[:States]
      maxi = NaN
      for prvstate in hmm[:States]
        temp = v[prvstate, k - 1] + log(hmm[:Trans_Probs][prvstate, state])
        maxi = maximum([maxi temp])
      end
      v[state, k] = log(pdf(hmm[:Emiss_Probs][state], obs[k])) + maxi
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
end
v

log(hmm[:Start_Probs][state] * pdf(hmm[:Emiss_Probs][state], obs[1]))
for state in hmm[:States]
  v[state, 1] = log(hmm[:Start_Probs][state] * hmm[:Emiss_Probs][state, obs[1]])
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
graph(sim_hmm[:Emission])



emiss_dist = hmm[:Emiss_Probs]

emissions = zeros(10)
states = repmat(["state"], 10)
states[1] = wsample(hmm[:States], hmm[:Start_Probs], 1)[1]
emissions[1] = rand(emiss_dist[states[1]], 1)[1]
hmm_seq = simHMM(hmm, 15)
isa(a, Dict)
isa(["C1", "C2"], AbstractArray)
AbstractString <: Tuple(AbstractString AbstractArray)
isa(Distributions.Normal(), Distributions.ContinuousDistribution)



#######
states = ["C1", "C2"]
symbols = ["H", "T"]
startprobs = [.7, .2]
emissprobs = Union{}
transprobs = Union{}
n_states = (states |> size)[1]
n_symbols = (symbols |> size)[1]

init_start = NamedArray(repmat([1 / n_states], n_states))
init_trans = NamedArray(.5 * eye(n_states) + repmat([0.5 / n_states], n_states, n_states))


  if emissprobs == Union{}
    init_emiss = Dict{AbstractString, Distributions.Distribution}()
    for i in states
      init_emiss[i] = Distributions.Normal()
    end
  elseif (emissprobs != Union{}) & (isa(emissprobs, Distributions.ContinuousDistribution) || isa(emissprobs, Distribution.DiscreteDistribution))
    init_emiss = emissprobs
  elseif (emissprobs != Union{}) & isa(emissprobs, AbstractVector)
    init_emiss = NamedArray(repmat([1 / n_symbols], n_states, n_symbols))

    setnames!(init_emiss, states, 1)
    setnames!(init_emiss, symbols, 2)

    setdimnames!(init_emiss, "States", 1)
    setdimnames!(init_emiss, "Symbols", 2)
  end
init_emiss
init_emiss = NamedArray(rand(emiss_dist(), (n_states, n_symbols)))

if startprobs != Union{}
  init_start = NamedArray(startprobs)
end
if transprobs != Union{}
  init_trans = NamedArray(transprobs)
end
if emissprobs != Union{}
  init_emiss = NamedArray(emissprobs)
end

setnames!(init_start, states, 1)
setnames!(init_trans, states, 1)
setnames!(init_trans, states, 2)
setnames!(init_emiss, states, 1)
setnames!(init_emiss, symbols, 2)

setdimnames!(init_start, "States", 1)
setdimnames!(init_trans, "from", 1)
setdimnames!(init_trans, "to", 2)
setdimnames!(init_emiss, "States", 1)
setdimnames!(init_emiss, "Symbols", 2)

return Dict(
  :States => states,
  :Symbols => symbols,
  :Start_Probs => init_start,
  :Trans_Probs => init_trans,
  :Emiss_Probs => init_emiss
  )





#######




function a1(y::AbstractArray, x::Function, params...)
  pdf(x(params...), y)
end

function pow (x, args...)
  x^args
end
rand(Normal(1, 1), 10)
pdf(Normal(1, 1), 10)
pdf(Normal(1, 1), 10)


pdf(Normal(1, 1), [1, 2])

pdf(Normal(), [1; 2; 3])
a1([1; 2; 3], Normal, 1, 2)

call(a2::Function, args...) = a2(args...)
a2 = Normal
pdf(a2(10, 1), [1; 2; 3])
a2(10, 1)
pdf(Normal(0), [1; 2; 3])
a2 = Normal
pdf(a2(), [1; 2; 3])
# Function for markov sequence
function genmarkovseq(states, trans_mat, init_probs, τ)

  mysequence = Array(AbstractString, 0)

  first_val = wsample(states, init_probs, 1)
  append!(mysequence, first_val)

  for i in 2:τ
    prev_val = mysequence[i - 1]
    probs = trans_mat[prev_val, :]

    new_val = wsample(states, vec(probs), 1)
    append!(mysequence, new_val)
  end

  return mysequence
end

# Example of generating a markov sequence
nucleotides = ["A"; "C"; "G"; "T"]
initialprobs = [0.25; 0.25; 0.25; 0.25]
mytransitionmatrix = NamedArray([[0.20 0.30 0.30 0.20];
                                 [0.10 0.41 0.39 0.10];
                                 [0.25 0.25 0.25 0.25];
                                 [0.50 0.17 0.17 0.17]])
setnames!(mytransitionmatrix, nucleotides, 1)
setnames!(mytransitionmatrix, nucleotides, 2)

genmarkovseq(nucleotides, mytransitionmatrix, initialprobs, 30)

function genhmmseq(obs_states, lat_states, trans_mat, emiss_mat, init_probs, τ)

  mysequence = Array(AbstractString, 0)
  mystates = Array(AbstractString, 0)

  first_state = wsample(lat_states, init_probs, 1)
  probs = emiss_mat[first_state, :]

  first_val = wsample(obs_states, vec(probs), 1)
  append!(mysequence, first_val)
  append!(mystates, first_state)

  for i in 2:τ
    prev_state = mystates[i - 1]
    state_prob = trans_mat[prev_state, :]
    state = wsample(lat_states, vec(state_prob), 1)

    probs = emiss_mat[state, :]
    new_val = wsample(obs_states, vec(probs), 1)

    append!(mysequence, new_val)
    append!(mystates, state)
  end

  return [mystates mysequence]
end

# Example of generating HMM sequence
nucleotides = ["A"; "C"; "G"; "T"]
states = ["AT-rich"; "GC-rich"]
initialprobs = [0.5; 0.5]
mytransitionmatrix = NamedArray([[0.7 0.3];
                                 [0.1 0.9]])
setnames!(mytransitionmatrix, states, 1)
setnames!(mytransitionmatrix, states, 2)
myemissionmatrix = NamedArray([[0.39 0.10 0.10 0.41];
                               [0.10 0.41 0.39 0.10]])
setnames!(myemissionmatrix, states, 1)
setnames!(myemissionmatrix, nucleotides, 2)
allnames(myemissionmatrix)[1]
mysequence = genhmmseq(nucleotides, states, mytransitionmatrix, myemissionmatrix, initialprobs, 30)
mysequence

function makeviterbiMat(sequence, trans_mat, emiss_mat)
  sequence = sequence |> uppercase
  numstates = (trans_mat |> size)[1]
  v = zeros(((sequence |> size)[1], numstates))
  v[1, :] = 0
  v[1, 1] = 1

  for i in 2:(sequence |> size)[1]
    for l in 1:numstates
      valprobs_statel = emiss_mat[l, sequence[i]]
      v[i, l] = valprobs_statel * maximum(v[i - 1, :] * trans_mat[:, l])
    end
  end

  return v
end


""" The following functions are taken from the source code of HMM R package
"""
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

function forward1(hmm, obs)
  """ FUNCTION FOR COMPUTING THE FORWARD PROBABILITY

  hmm - object hmm
  obs - the observations given
  """
  n_obs = (obs |> size)[1]
  n_states = (hmm[:States] |> size)[1]
  f = NamedArray(zeros((n_states, n_obs)))
  setnames!(f, hmm[:States], 1)

  setdimnames!(f, "states", 1)
  setdimnames!(f, "index", 2)

  # Initialize
  for state in hmm[:States]
    f[state, 1] = log(hmm[:Start_Probs][state] * hmm[:Emiss_Probs][state, obs[1]])
  end

  # Iteration
  for k in 2:n_obs
    for state in hmm[:States]
      logsum = -Inf
      for prvstate in hmm[:States]
        temp = f[prvstate, k - 1] + log(hmm[:Trans_Probs][prvstate, state])
        if temp > -Inf
          logsum = temp + log(1 + exp(logsum - temp))
        end
      end
      f[state, k] = log(hmm[:Emiss_Probs][state, obs[k]]) + logsum
    end
  end

  return f |> exp
end

function backward1(hmm, obs)
  """ FUNCTION FOR COMPUTING THE BACKWARD PROBABILITY

  hmm - object hmm
  obs - the observations given
  """
  n_obs = (obs |> size)[1]
  n_states = (hmm[:States] |> size)[1]
  b = NamedArray(zeros((n_states, n_obs)))
  setnames!(b, hmm[:States], 1)
  setdimnames!(b, "states", 1)
  setdimnames!(b, "index", 2)

  # Initialize
  for state in hmm[:States]
    b[state, n_obs] = log(1)
  end

  # Iteration
  for k in collect((n_obs - 1): -1 : 1)
    for state in hmm[:States]
      logsum = -Inf
      for nxtstate in hmm[:States]
        temp = b[nxtstate, k + 1] + log(hmm[:Trans_Probs][state, nxtstate] *
          hmm[:Emiss_Probs][nxtstate, obs[k + 1]])
        if temp > -Inf
          logsum = temp + log(1 + exp(logsum - temp))
        end
      end
      b[state, k] = logsum
    end
  end

  return b |> exp
end


isa(["C1", "C2"], AbstractArray)

isa([.7; .2], AbstractArray)
AbstractString :>
AbstractArray <: Real
AbstractFloat <: Real

AbstractString <: {AbstractArray, AbstractString}
hmm = initHMM(["C1", "C2"], ["H", "T"], [.7, .2])
hmm
hmm_seq = simHMM(hmm, 15)
hmm_seq[:Emission]
z = forward(hmm, hmm_seq[:Emission]) .* backward(hmm, hmm_seq[:Emission])
sum(z, 1)
obs = hmm_seq[:Emission]
n_obs = (obs |> size)[1]
n_states = (hmm[:States] |> size)[1]
f = NamedArray(zeros((n_states, n_obs)))
setnames!(f, hmm[:States], 1)

setdimnames!(f, "states", 1)
setdimnames!(f, "index", 2)

# Initialize
for state in hmm[:States]
  f[state, 1] = log(hmm[:Start_Probs][state] * hmm[:Emiss_Probs][state, obs[1]])
end


f



hmm_seq[:]
vec(hmm[:Start_Probs])
hmm[:Start_Probs]["C1"]
vec(hmm[:Trans_Probs][states[1], :])
states   = Array(AbstractString, 0)
emission = Array(AbstractString, 0)
states[1]
append!(states, wsample(hmm[:States], vec(hmm[:Start_Probs]), 1))
i = 4
append!(states, wsample(hmm[:States], vec(hmm[:Trans_Probs][states[i - 1], :]), 1))
j = 1
append!(emission, wsample(hmm[:Symbols], vec(hmm[:Emiss_Probs][states[j], :]), 1))
obs =
n_obs = (obs |> size)[1]
n_states = (hmm[:States] |> size)[1]
f = NamedArray(Array(AbstractVector, (n_states, n_obs)))
setnames!(f, hmm[:States], 1)
setnames!(f, collect(1:n_obs), 1)
setdimnames!(f, "states", 1)
setdimnames!(f, "index", 2)



function f1(N::Int)
    x = Array(Int, N)
    for n = 1:N
        x[n] = n
    end
    return(x)
end

function f2(N::Int)
    x = Array(Int, 0)
    for n = 1:N
        push!(x, n)
    end
    return(x)
end

f1(2)
f2(2)

N = 50000
@time f1(N)
@time f2(N)
