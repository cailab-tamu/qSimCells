library(dplyr)
library(Seurat)
library(ggplot2)
library(Matrix)
library(hdf5r)
library(presto)
library(ggpubr)
library(tidyr)
library(stringr)
library(tibble)
library(cowplot)
library(openxlsx)
library(patchwork)
library(CellChat)
library(readxl)

set.seed(123)

# --- SETUP AND DATA PROCESSING -----------------------------------------------
base_dir <- "C:\\Users\\selim\\Documents\\vs_working_dir\\qSimCells\\r_cellchat_qsim"
setwd(base_dir)

data <- readRDS(file.path(base_dir, "sim_merged_datasets_co_mo.rds"))
table(data$CellType, data$BatchID)

process_rna <- function(data, assay_name = "RNA", num_hvg = 2000,
                        dims_pca = 50, resolution = 1.0) {
  DefaultAssay(data) <- assay_name
  data <- FindVariableFeatures(data, selection.method = "vst", nfeatures = num_hvg)
  data <- ScaleData(data)
  data <- RunPCA(data)
  data <- RunUMAP(data, dims = 1:dims_pca, n.epochs = 500)
  data <- FindNeighbors(data, dims = 1:dims_pca)
  data <- FindClusters(data, resolution = resolution)
  return(data)
}

data <- process_rna(data, dims_pca = 10)

print(DimPlot(object = data, reduction = "umap",
              group.by = c("CellType", "BatchID"),
              label = TRUE, repel = TRUE, label.size = 5,
              label.box = TRUE, alpha = 1, raster = FALSE, pt.size = 2) +
        NoLegend())

# --- CONDITION METADATA ------------------------------------------------------
data$Condition <- as.character(data$BatchID)
data$Condition[grepl("Co", data$BatchID)] <- "Co"
data$Condition[grepl("Mo", data$BatchID)] <- "Mo"
data$samples <- factor(data$Condition)

data_co <- subset(data, Condition == "Co")
data_co$CellType <- factor(data_co$CellType)

data_mo <- subset(data, Condition == "Mo")
data_mo$CellType <- factor(data_mo$CellType)

# --- CELLCHAT OBJECTS --------------------------------------------------------
cellchat_Mo <- createCellChat(object = data_mo, meta = data_mo@meta.data,
                              group.by = "CellType", assay = "RNA")
cellchat_Co <- createCellChat(object = data_co, meta = data_co@meta.data,
                              group.by = "CellType", assay = "RNA")

# --- CUSTOM DATABASE ---------------------------------------------------------
qsim_db     <- list()
sheet_names <- excel_sheets("qsimDB.xlsx")
for (sheet in sheet_names) {
  qsim_db[[sheet]] <- read_excel("qsimDB.xlsx", sheet = sheet)
}
print(head(qsim_db$interaction, 5))

cellchat_Mo@DB <- qsim_db
cellchat_Co@DB <- qsim_db

# --- CELLCHAT INFERENCE — Mo -------------------------------------------------
cellchat_Mo <- setIdent(cellchat_Mo, ident.use = "CellType")
cellchat_Mo <- subsetData(cellchat_Mo, features = rownames(cellchat_Mo@data))
cellchat_Mo <- identifyOverExpressedGenes(cellchat_Mo)
cellchat_Mo <- identifyOverExpressedInteractions(cellchat_Mo)
cellchat_Mo <- computeCommunProb(cellchat_Mo, type = "truncatedMean", trim = 0.01, nboot = 1000)
cellchat_Mo <- computeCommunProbPathway(cellchat_Mo, thresh = 0.05)
cellchat_Mo <- aggregateNet(cellchat_Mo)

# --- CELLCHAT INFERENCE — Co -------------------------------------------------
cellchat_Co <- setIdent(cellchat_Co, ident.use = "CellType")
cellchat_Co <- subsetData(cellchat_Co, features = rownames(cellchat_Co@data))
cellchat_Co <- identifyOverExpressedGenes(cellchat_Co)
cellchat_Co <- identifyOverExpressedInteractions(cellchat_Co)
cellchat_Co <- computeCommunProb(cellchat_Co, type = "truncatedMean", trim = 0.01, nboot = 1000)
cellchat_Co <- computeCommunProbPathway(cellchat_Co, thresh = 0.05)
cellchat_Co <- aggregateNet(cellchat_Co)

# --- MERGE -------------------------------------------------------------------
cellchat_merged <- mergeCellChat(list(Mo = cellchat_Mo, Co = cellchat_Co),
                                 add.names = c("Mo", "Co"))

# --- Step 1: Run cross-condition DE with relaxed thresholds ------------------
cellchat_merged <- identifyOverExpressedGenes(
  cellchat_merged,
  group.dataset = "datasets",
  pos.dataset   = "Co",
  features.name = "differential_genes_relaxed",
  only.pos      = FALSE,
  thresh.pc     = 0.0,
  thresh.fc     = 0.0,
  thresh.p      = 1.0
)





# --- BUILD BASE NETWORK ------------------------------------------------------
# Bind the flat interaction tables from the list into a single data frame
net_list <- subsetCommunication(cellchat_merged)
net <- bind_rows(net_list, .id = "datasets") 

# --- Step 2: Build streamlined lookup table ----------------------------------
feat_info <- cellchat_merged@var.features[["differential_genes_relaxed.info"]]

# Use CellChat's native 'datasets' column directly
feat_lookup <- feat_info %>%
  select(clusters, features, logFC, pvalues, pct.1, pct.2, datasets) %>%
  distinct(clusters, features, .keep_all = TRUE)

# --- Step 3: Join DE statistics safely using dplyr ---------------------------
net_manual <- net %>%
  # Join for Ligand
  left_join(feat_lookup, by = c("source" = "clusters", "ligand" = "features")) %>%
  rename(ligand.logFC = logFC, ligand.pvalues = pvalues, ligand.pct.1 = pct.1, 
         ligand.pct.2 = pct.2, ligand.upregulated_in = datasets.y) %>%
  rename(datasets = datasets.x) %>% 
  # Join for Receptor
  left_join(feat_lookup, by = c("target" = "clusters", "receptor" = "features")) %>%
  rename(receptor.logFC = logFC, receptor.pvalues = pvalues, receptor.pct.1 = pct.1, 
         receptor.pct.2 = pct.2, receptor.upregulated_in = datasets.y) %>%
  rename(datasets = datasets.x)

# --- Step 4: Calculate Co/Mo Probability Ratios Accurately -------------------
# Reshape base communication probabilities side-by-side to cleanly compute ratios
prob_ratios <- net %>%
  select(source, target, ligand, receptor, datasets, prob) %>%
  distinct(source, target, ligand, receptor, datasets, .keep_all = TRUE) %>% 
  tidyr::pivot_wider(names_from = datasets, values_from = prob, values_fill = 0) %>%
  mutate(prob_ratio_Co_Mo = Co / Mo) %>%
  select(source, target, ligand, receptor, prob_ratio_Co_Mo)

# Merge ratios back into the main manual annotation table
net_manual <- net_manual %>%
  left_join(prob_ratios, by = c("source", "target", "ligand", "receptor"))

# --- Step 5: Format and Export -----------------------------------------------
cols_to_round <- c("prob", "ligand.logFC", "ligand.pvalues", "ligand.pct.1", "ligand.pct.2",
                   "receptor.logFC", "receptor.pvalues", "receptor.pct.1", "receptor.pct.2",
                   "prob_ratio_Co_Mo")

net_print <- net_manual %>%
  mutate(across(all_of(cols_to_round), ~ round(.x, 3)))

# View clean snippet with expression p-values and natural dataset track labels
print(head(net_print[, c("source", "target", "ligand", "receptor",
                         "prob", "pval", "datasets", "prob_ratio_Co_Mo",
                         "ligand.logFC", "ligand.pvalues", "ligand.upregulated_in",
                         "receptor.logFC", "receptor.pvalues", "receptor.upregulated_in")]))

write.csv(feat_lookup, "de_genes_Co_vs_Mo.csv", row.names = FALSE)
write.csv(net_manual,  "cellchat_net_with_DE.csv", row.names = FALSE)


# --- VISUALIZATIONS ----------------------------------------------------------
weight.max <- getMaxWeight(list(cellchat_Co, cellchat_Mo), attribute = c("count", "weight"))

par(mfrow = c(1, 2), xpd = TRUE)
netVisual_circle(cellchat_Co@net$count, weight.scale = TRUE, label.edge = FALSE,
                 edge.weight.max = weight.max[1], title.name = "Number of interactions - Co")
netVisual_circle(cellchat_Mo@net$count, weight.scale = TRUE, label.edge = FALSE,
                 edge.weight.max = weight.max[1], title.name = "Number of interactions - Mo")

netVisual_bubble(cellchat_merged,
                 sources.use = c("CellType1", "CellType2"),
                 targets.use = c("CellType1", "CellType2"),
                 comparison  = c(1, 2),
                 angle.x     = 45)