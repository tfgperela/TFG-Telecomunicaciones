library(plspm)
#library(ggplot2)
mydata = read.csv("/home/rodrigo/Documentos/tfg-teleco/TFG-Telecomunicaciones/indices/todo/ad_0.csv", header = TRUE)

head(mydata, n = 5)

Parasimpatico = c(0, 0, 0, 0)
Simpatico = c(0, 0, 0, 0)
ANS = c(1, 1, 0, 0)
Recall = c(0, 0, 1, 0)
ad_path = rbind(Parasimpatico, Simpatico, ANS, Recall)
colnames(ad_path) = rownames(ad_path)

innerplot(ad_path)

#Higher-order constructs in PLS-PM: 1) Repeated Indicators Approach
#Posible interesante usar 3) Hybrid Approach
ad_blocks = list(c(5, 6, 9), c(2, 3, 7, 8), c(5, 6, 9, 2, 3, 7, 8), 11)
#Los indicadores de "EDA" y "Score" son formative ("B")
ad_modes = c("A", "A", "A", "A")

ad_pls = plspm(mydata, ad_path, ad_blocks, modes = ad_modes)
outerplot(ad_pls)
summary(ad_pls)
    