library(plspm)
#library(ggplot2)
mydata = read.csv("/home/rodrigo/Documentos/tfg-teleco/TFG-Telecomunicaciones/indices/todo/prueba/ad_0_mujeres.csv", header = TRUE)

head(mydata, n = 5)

Vagal = c(0, 0, 0, 0)
Simpatico = c(0, 0, 0, 0)
ANS = c(1, 1, 0, 0)
Recall = c(0, 0, 1, 0)
ad_path = rbind(Vagal, Simpatico, ANS, Recall)
colnames(ad_path) = rownames(ad_path)

innerplot(ad_path)

#Higher-order constructs in PLS-PM: 1) Repeated Indicators Approach
#Posible interesante usar 3) Hybrid Approach
ad_blocks = list(c(5, 6, 9), c(2, 7, 8, 12), c(3,10), 11)
#Los indicadores de "EDA" y "Score" son formative ("B")
ad_modes = c("A", "A", "A", "A")

#ad_pls = plspm(mydata, ad_path, ad_blocks, modes = ad_modes)
ad_pls = plspm(mydata, ad_path, ad_blocks, modes = ad_modes, maxiter = 500)
plot(ad_pls)
#plot(ad_pls, what = "loadings", arr.width = 0.1)
outerplot(ad_pls)
summary(ad_pls)
    
head(ad_pls$scores, n = 5)
      
#Measurement Model Assessment
ad_pls$unidim
plot(ad_pls, what = "loadings")
ad_pls$outer_model
ad_pls$crossloadings

#Structural Model Assessment
ad_pls$inner_model
ad_pls$inner_summary
ad_pls$gof

