<font color = "#645452">Chapter 3</font>
================

by **Al-Ahmadgaid B. Asaad** (`alstatr.blogspot.com`; `alasaadstat@gmail.com`/`alstated@gmail.com`). This notebook contains source codes used in the thesis.

<table style="width:136%;">
<colgroup>
<col width="19%" />
<col width="116%" />
</colgroup>
<thead>
<tr class="header">
<th align="left"><code>Chapter Title</code></th>
<th align="left"><font color = "#FFA700">Bayesian Inference and Basic Definitions </font></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left"><code><strong>Thesis Title</strong></code></td>
<td align="left"><font color = "#FFA700"><strong>Bayesian Inference of Artificial Neural Networks and Hidden Markov Models</strong></font></td>
</tr>
</tbody>
</table>

Slide with Bullets
------------------

-   Bullet 1
-   Bullet 2
-   Bullet 3

Slide with R Output
-------------------

``` r
library(animation)

saveGIF({
  for(i in 1:100){
    curve(sin(x), from = -5 + (i * 0.05), to = 5 + (i * 0.05), col = "red", ylab = "")
    curve(cos(x), from = -5 + (i * 0.05), to = 5 + (i * 0.05), add = TRUE, col = "blue", ylab = "")
    legend("topright", legend = c("sin(x)", "cos(x)"), fill = c("red", "blue"), bty = "n")
  }
}, interval = 0.1, ani.width = 550, ani.height = 350)
```

    ## Executing: 
    ## ""convert" -loop 0 -delay 10 Rplot1.png Rplot2.png Rplot3.png
    ##     Rplot4.png Rplot5.png Rplot6.png Rplot7.png Rplot8.png
    ##     Rplot9.png Rplot10.png Rplot11.png Rplot12.png Rplot13.png
    ##     Rplot14.png Rplot15.png Rplot16.png Rplot17.png Rplot18.png
    ##     Rplot19.png Rplot20.png Rplot21.png Rplot22.png Rplot23.png
    ##     Rplot24.png Rplot25.png Rplot26.png Rplot27.png Rplot28.png
    ##     Rplot29.png Rplot30.png Rplot31.png Rplot32.png Rplot33.png
    ##     Rplot34.png Rplot35.png Rplot36.png Rplot37.png Rplot38.png
    ##     Rplot39.png Rplot40.png Rplot41.png Rplot42.png Rplot43.png
    ##     Rplot44.png Rplot45.png Rplot46.png Rplot47.png Rplot48.png
    ##     Rplot49.png Rplot50.png Rplot51.png Rplot52.png Rplot53.png
    ##     Rplot54.png Rplot55.png Rplot56.png Rplot57.png Rplot58.png
    ##     Rplot59.png Rplot60.png Rplot61.png Rplot62.png Rplot63.png
    ##     Rplot64.png Rplot65.png Rplot66.png Rplot67.png Rplot68.png
    ##     Rplot69.png Rplot70.png Rplot71.png Rplot72.png Rplot73.png
    ##     Rplot74.png Rplot75.png Rplot76.png Rplot77.png Rplot78.png
    ##     Rplot79.png Rplot80.png Rplot81.png Rplot82.png Rplot83.png
    ##     Rplot84.png Rplot85.png Rplot86.png Rplot87.png Rplot88.png
    ##     Rplot89.png Rplot90.png Rplot91.png Rplot92.png Rplot93.png
    ##     Rplot94.png Rplot95.png Rplot96.png Rplot97.png Rplot98.png
    ##     Rplot99.png Rplot100.png "animation.gif""

    ## Output at: animation.gif

    ## [1] TRUE

Slide with Plot
---------------

``` r
plot(pressure)
```

![](sample_files/figure-markdown_github/pressure-1.png)<!-- -->
