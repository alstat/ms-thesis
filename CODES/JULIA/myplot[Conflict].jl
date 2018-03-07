#module cool

using PyPlot

col, bgcol, grdcol, lty, ticksize = ("#E9D66B", "", "#696969", "dotted", 9)

function xyplot(x::AbstractVector, y::AbstractVector; xlab::AbstractString = "", ylab::AbstractString  = "", color::AbstractString = col, bgcol::AbstractString = bgcol, marker::AbstractString = "", markersize::Int = 5)
    matplotlib[:rc]("font", family = "Times New Roman")
    matplotlib[:rc]("xtick", direction = "out")
    matplotlib[:rc]("ytick", direction = "out")
    """ X-Y PLOT
    --------------------------
    x - input for x-axis
    y - input for y-axis
    """
    f, ax = subplots(figsize = (5, 5))
    ax[:plot](x, y, color = color, marker = marker, markersize = markersize)
    ax[:set_xlabel](xlab)
    ax[:set_ylabel](ylab)
    ax[:grid]("on", which = "major", color = grdcol, linestyle = lty)
    ax[:set_axis_bgcolor](bgcol)
    ax[:tick_params](axis = "both", which = "major", labelsize = ticksize, pad = 5)
    ax[:tick_params](axis = "both", which = "minor", labelsize = ticksize, pad = 5)
    ax[:set_axisbelow]("on")
    ax[:locator_params](axis = "x", nbins = 8, tight = true)
    ax[:locator_params](axis = "y", nbins = 8, tight = true)
    ax[:margins](.03)
end

n = 100
x = y = linspace(-4, 5, n)
z = zeros((n, n))

xgrid = repmat(x, 1, n)
ygrid = repmat(y', n, 1)

for i in 1:n
  for j in 1:n
    z[i:i, j:j] = gmm([x[i]; y[j]])
  end
end

n = 1000
d = sim_gmm(n)
xyplot(d'[:, 1], d'[:, 2], marker = "o")


function graph(x, y = Union{}; xlab = "", ylab = "", color = col, marker = "", markersize = 5,
    ptype = "xyplot", where = "post", width = 1, bins = 30)
    if (y == Union{}) & (ptype != "hist")
        y = x
        ylen = y |> length
        x = collect(1:ylen)

        ylim_margin = -(y |> maximum, y |> minimum) * .05
        xlim_margin = -(x |> maximum, x |> minimum) * .05
    elseif (y == Union{}) & (ptype == "hist")
        y = x
        ylim_margin = -(y |> maximum, y |> minimum) * .05
        xlim_margin = -(x |> maximum, x |> minimum) * .05
    else
        ylim_margin = -(y |> maximum, y |> minimum) * .05
        xlim_margin = -(x |> maximum, x |> minimum) * .05
    end

    f, ax = subplots(figsize = (5, 5))

    if ptype == "step"
        ax[:step](x, y, where = where, color = color, marker = marker, markersize = markersize)
    elseif ptype == "xyplot"
        ax[:plot](x, y, color = color, marker = marker, markersize = markersize)
    elseif ptype == "bar"
        ax[:bar](x, y, width, color = color)
    elseif ptype == "hist"
        ax[:hist](x, bins, edgecolor = "white", facecolor = "black")
    end

    ax[:set_xlabel](xlab)
    ax[:set_ylabel](ylab)
    ax[:grid]("on", which = "major", color = grdcol, linestyle = lty)
    ax[:set_axis_bgcolor](bgcol)
    ax[:set_ylim](((y |> minimum) - ylim_margin, (y |> maximum) + ylim_margin))
    ax[:set_xlim](((x |> minimum) - xlim_margin, (x |> maximum) + xlim_margin))
    ax[:tick_params](axis = "both", which = "major", labelsize = ticksize)
    ax[:tick_params](axis = "both", which = "minor", labelsize = ticksize)
    ax[:set_axisbelow]("on")
end
#end
