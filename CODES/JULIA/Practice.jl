cd(joinpath(homedir(), "Dropbox/Julia Project"))
using myplot


x = collect(-4:.001:4)
y = sin(x)


plot(x, y, "-")
