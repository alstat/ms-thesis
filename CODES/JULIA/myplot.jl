module myplot

using PyPlot

function xyplot(x::AbstractVector, y::AbstractVector; kwargs...)
  f, ax = subplots(figsize = (5, 5))
  ax[:plot](x, y, kwargs...)
end

x = linspace(-5, 5, 100)
y = x |> sin

xyplot(x, y)
col, bgcol, grdcol, lty, ticksize = ("black", "white", "#848482", "dotted", 8)

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
end
end
