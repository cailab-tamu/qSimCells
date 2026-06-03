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
cellchat_Mo <- computeCommunProb(cellchat_Mo, type = "truncatedMean", trim = 0.01)
cellchat_Mo <- computeCommunProbPathway(cellchat_Mo, thresh = 0.05)
cellchat_Mo <- aggregateNet(cellchat_Mo)

# --- CELLCHAT INFERENCE — Co -------------------------------------------------
cellchat_Co <- setIdent(cellchat_Co, ident.use = "CellType")
cellchat_Co <- subsetData(cellchat_Co, features = rownames(cellchat_Co@data))
cellchat_Co <- identifyOverExpressedGenes(cellchat_Co)
cellchat_Co <- identifyOverExpressedInteractions(cellchat_Co)
cellchat_Co <- computeCommunProb(cellchat_Co, type = "truncatedMean", trim = 0.01)
cellchat_Co <- computeCommunProbPathway(cellchat_Co, thresh = 0.05)
cellchat_Co <- aggregateNet(cellchat_Co)

# --- MERGE -------------------------------------------------------------------
cellchat_merged <- mergeCellChat(list(Mo = cellchat_Mo, Co = cellchat_Co),
                                 add.names = c("Mo", "Co"))

# --- BUILD net: full communication table from merged object ------------------
# subsetCommunication extracts the flat interaction table (prob + pval)
# across both conditions — this is the base table we annotate with DE results.
net <- subsetCommunication(cellchat_merged)

# =============================================================================
# MANUAL DE MAPPING TO CELLCHAT NETWORK
# Bypasses netMappingDEG which fails on merged objects due to dataset-pooling.
# Uses cell-type-aware Co vs Mo DE results (identifyOverExpressedGenes with
# relaxed thresholds) and joins directly onto the communication table.
# =============================================================================

# --- Step 1: Run cross-condition DE with relaxed thresholds ------------------
# thresh.pc=0, thresh.fc=0, thresh.p=1 ensures all quantum genes (g0-g9)
# are included regardless of sparsity — necessary because quantum genes are
# binary (ON/OFF) and fail standard percent-expressed filters.
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

# --- Step 2: Build lookup table ----------------------------------------------
# features.info contains per-gene Co vs Mo DE statistics,
# stratified by cell type (clusters column).
feat_info      <- cellchat_merged@var.features[["differential_genes_relaxed.info"]]
feat_info$key  <- paste(feat_info$clusters, feat_info$features, sep = ".")
feat_info_dedup <- feat_info[!duplicated(feat_info$key), ]

# --- Step 3: Join DE statistics onto communication table ---------------------
# pval         = CellChat permutation p-value for communication probability
#                (0 means p < 1/N_permutations, i.e., p < 0.01)
# ligand/receptor.pvalues = Wilcoxon DE p-value (Co vs Mo)
# ligand/receptor.logFC   = log2 fold-change (Co vs Mo)
net_manual <- net

# Build join keys from source+ligand and target+receptor
net_flat$source.ligand   <- paste(net_flat$source, net_flat$ligand,   sep = ".")
net_flat$target.receptor <- paste(net_flat$target, net_flat$receptor, sep = ".")

# Now join
net_manual <- net_flat
net_manual$ligand.logFC   <- feat_info_dedup$logFC[match(net_manual$source.ligand,   feat_info_dedup$key)]
net_manual$ligand.pvalues <- feat_info_dedup$pvalues[match(net_manual$source.ligand, feat_info_dedup$key)]
net_manual$ligand.pct.1   <- feat_info_dedup$pct.1[match(net_manual$source.ligand,   feat_info_dedup$key)]
net_manual$ligand.pct.2   <- feat_info_dedup$pct.2[match(net_manual$source.ligand,   feat_info_dedup$key)]
net_manual$receptor.logFC   <- feat_info_dedup$logFC[match(net_manual$target.receptor,   feat_info_dedup$key)]
net_manual$receptor.pvalues <- feat_info_dedup$pvalues[match(net_manual$target.receptor, feat_info_dedup$key)]
net_manual$receptor.pct.1   <- feat_info_dedup$pct.1[match(net_manual$target.receptor,   feat_info_dedup$key)]
net_manual$receptor.pct.2   <- feat_info_dedup$pct.2[match(net_manual$target.receptor,   feat_info_dedup$key)]

# Add Co/Mo probability ratio
prob_mo <- net_manual$prob[net_manual$datasets == "Mo"]
prob_co <- net_manual$prob[net_manual$datasets == "Co"]
net_manual$prob_ratio_Co_Mo <- NA
net_manual$prob_ratio_Co_Mo[net_manual$datasets == "Mo"] <- prob_co / prob_mo
net_manual$prob_ratio_Co_Mo[net_manual$datasets == "Co"] <- prob_co / prob_mo

# Round numeric columns to 3 decimal places for display
cols_to_round <- c("prob", "ligand.logFC", "ligand.pvalues", "ligand.pct.1", "ligand.pct.2",
                   "receptor.logFC", "receptor.pvalues", "receptor.pct.1", "receptor.pct.2",
                   "prob_ratio_Co_Mo")
net_print <- net_manual
net_print[, cols_to_round] <- lapply(net_print[, cols_to_round], round, digits = 3)

print(net_print[, c("source", "target", "ligand", "receptor",
                    "prob", "pval", "datasets", "prob_ratio_Co_Mo",
                    "ligand.logFC",   "ligand.pvalues",
                    "receptor.logFC", "receptor.pvalues")])

write.csv(feat_info_dedup, "de_genes_Co_vs_Mo.csv",   row.names = TRUE)
write.csv(net_manual,      "cellchat_net_with_DE.csv", row.names = FALSE)


# --- VISUALIZATIONS ----------------------------------------------------------
weight.max <- getMaxWeight(list(cellchat_Co, cellchat_Mo),
                           attribute = c("count", "weight"))
par(mfrow = c(1, 2), xpd = TRUE)
netVisual_circle(cellchat_Co@net$count, weight.scale = TRUE, label.edge = FALSE,
                 edge.weight.max = weight.max[1],
                 title.name = "Number of interactions - Co")
netVisual_circle(cellchat_Mo@net$count, weight.scale = TRUE, label.edge = FALSE,
                 edge.weight.max = weight.max[1],
                 title.name = "Number of interactions - Mo")

netVisual_bubble(cellchat_merged,
                 sources.use = c("CellType1", "CellType2"),
                 targets.use = c("CellType1", "CellType2"),
                 comparison  = c(1, 2),
                 angle.x     = 45)
