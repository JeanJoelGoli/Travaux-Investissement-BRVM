# data_BRVM.R

library(BRVM)
library(remotes)
library(arrow)

capit_init <- BRVM_cap()
capit_ <- as.data.frame(capit_init)

# 3) Sauvegarde
write.csv(capit_, "60_Capitalisations.csv", row.names = FALSE)

#message("Écrit en CSV (fallback)")
