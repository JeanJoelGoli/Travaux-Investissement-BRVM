# data_BRVM.R

library(BRVM)
library(remotes)
library(arrow)
library(dplyr)
library(stringr)
library(scales)
library(readr)


# 1) Métadonnées + liste des tickers
desc <- BRVM_ticker_desc()
symbols <- desc$Ticker
# Pour tester plus vite : symbols <- head(symbols, 20)

# 2) Données marché
#rich <- BRVM_get(.symbol = symbols, .from = "1998-01-01", .to = Sys.Date())
#sika <- BRVM_get1("ALL INDEXES", Period = 0, from = "1998-01-01", to = Sys.Date())

rich <- BRVM_get(.symbol = symbols, .from = "1998-01-01", .to = Sys.Date() )
sika <- BRVM_get1("ALL INDEXES", Period = 0, from = "1998-01-01", to = Sys.Date() )
#capit_init <- BRVM_cap()
#capit_ <- as.data.frame(capit_init)


# 1) Métadonnées + liste des tickers

capi <- read.csv("/Users/jeanjoelgoli/Documents/FINANCE/Travaux BRVM/61_Capitalisations.csv")
symbols <- desc$Ticker


# --- 0) Nettoyer les noms de colonnes : espaces, NBSP, apostrophes ---
clean_names_soft <- function(nms) {
  nms %>%
    str_replace_all("\u00A0", " ") %>%   # NBSP -> espace
    str_replace_all("[’‘]", "'") %>%     # apostrophes typographiques -> '
    str_squish()                         # trim + compresser espaces
}

names(capi) <- clean_names_soft(names(capi))
names(desc) <- clean_names_soft(names(desc))
names(sika) <- clean_names_soft(names(sika))

capi2 <- capi %>%
  mutate(
    Number.of.shares          = parse_number(as.character(Number.of.shares),
                                             locale = locale(grouping_mark = " ", decimal_mark = ".")),
    Daily.price               = parse_number(as.character(Daily.price),
                                             locale = locale(grouping_mark = " ", decimal_mark = ".")),
    Floating.Capitalization   = parse_number(as.character(Floating.Capitalization),
                                             locale = locale(grouping_mark = " ", decimal_mark = ".")),
    Global.capitalization     = parse_number(as.character(Global.capitalization),
                                             locale = locale(grouping_mark = " ", decimal_mark = ".")),
    `Global.capitalization(%)`= parse_number(as.character(Global.capitalization....),
                                             locale = locale(grouping_mark = " ", decimal_mark = "."))
  ) %>%
  mutate(
    rang_capi = dense_rank(desc(Global.capitalization))
  )

# --- 2) Créer une palette de couleurs stable pour les tickers ---
unique_tickers <- unique(desc$Ticker)

# Exemple avec une palette qualitative (scales::hue_pal())
brvm_colors <- setNames(
  hue_pal()(length(unique_tickers)),
  unique_tickers
)

# --- 3) Ajouter la colonne color dans desc ---
desc2 <- desc %>%
  mutate(color = brvm_colors[Ticker])


# --- jointure desc -> sika
sika_plus <- rich %>%
  left_join(desc2, by = "Ticker")

# --- jointure capi -> sika
sika_final <- sika_plus %>%
  left_join(
    capi2 %>%
      select(Symbol,
             Global.capitalization, rang_capi,
             Number.of.shares, `Global.capitalization(%)`),
    by = c("Ticker" = "Symbol")
  ) %>%
  rename(
    "Global capitalization"      = "Global.capitalization",
    "Number of shares"           = "Number.of.shares",
    "Global capitalization (%)"  = "Global.capitalization(%)"
  )

# --- supprimer colonnes parasites éventuelles créées par la jointure
sika_final <- sika_final %>%
  select(-matches("\\.y$")) %>%  # enlève toutes les colonnes finissant par .y
  rename_with(~ gsub("\\.x$", "", .x)) # et nettoie les .x


# 3) Sauvegarde
write.csv(
  sika_final, 
  file = "/Users/jeanjoelgoli/Documents/FINANCE/Travaux BRVM/Cours_BRVM.csv", 
  row.names = FALSE
)
write.csv(
  rich, 
  file = "/Users/jeanjoelgoli/Documents/FINANCE/Travaux BRVM/60_Cours_indices.csv", 
  row.names = FALSE
)
