library(dplyr)
library(Seurat)
library(ggplot2)
library(Matrix)
library(hdf5r)
library(dittoSeq)
library(SplineDV)
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

# --- SETUP AND DATA PROCESSING ---
base_dir <- "C:\\Users\\ssromerogon\\Documents\\vscode_working_dir\\QuantumXCT\\python\\r_cellchat_qsim"
# base_dir <- "/mnt/SCDC/Optimus/selim_working_dir/2023_nr4a1_colon/results"
setwd(base_dir)

# Read Seurat object
data <- readRDS(file.path(base_dir, "sim_merged_datasets_co_mo.rds"))
table(data$CellType, data$BatchID)

# Function to process Seurat object
process_rna <- function(data, assay_name = "RNA", num_hvg = 2000, dims_pca = 50, resolution = 1.0) {
  DefaultAssay(data) <- assay_name
  data <- FindVariableFeatures(data, selection.method = "vst", nfeatures = num_hvg)
  data <- ScaleData(data)
  data <- RunPCA(data)
  data <- RunUMAP(data, dims = 1:dims_pca, n.epochs = 500)
  data <- FindNeighbors(data, dims = 1:dims_pca)
  data <- FindClusters(data, resolution = resolution)
  return(data)
}

data <- process_rna(data, dims_pca=10)

# DimPlot visualization
plot <- DimPlot(object = data, reduction = "umap", group.by = c("CellType"),
                label = TRUE, repel = TRUE, label.size = 3, label.box = TRUE, alpha = 1, raster=FALSE) +
  NoLegend()
print(plot)

# Create 'Condition' metadata column
data$Condition <- as.character(data$BatchID)
data$Condition[grepl("Co", data$BatchID)] <- "Co"
data$Condition[grepl("Mo", data$BatchID)] <- "Mo"
data$samples <- factor(data$Condition)

# Subset data and create CellChat objects
data_co <- subset(data, Condition == 'Co')
data_co$CellType <- factor(data_co$CellType)
table(data_co$CellType)

data_mo <- subset(data, Condition == 'Mo')
data_mo$CellType <- factor(data_mo$CellType)
table(data_mo$CellType)

cellchat_Mo <- createCellChat(object = data_mo, meta = data_mo@meta.data, group.by = "CellType", assay = "RNA")
cellchat_Co <- createCellChat(object = data_co, meta = data_co@meta.data, group.by = "CellType", assay = "RNA")

#rm(data_mo, data_co, data)
library(readxl)
# Assuming your Excel file is named 'qsimDB.xlsx' and is in the same directory.
file_name <- "qsimDB.xlsx"
# Initialize an empty list to hold the data frames
qsim_db <- list()
sheet_names <- excel_sheets(file_name)
# Loop through each sheet and load its data into a data frame
for (sheet in sheet_names) {
  cat(paste("Loading sheet:", sheet, "\n"))
  # read_excel automatically handles the header row.
  df <- read_excel(file_name, sheet = sheet)
  qsim_db[[sheet]] <- df
}
print(head(qsim_db$interaction,5))


CellChatDB <- qsim_db
cellchat_Mo@DB <- CellChatDB
cellchat_Co@DB <- CellChatDB

cellchat_Mo <- setIdent(cellchat_Mo, ident.use = "CellType")
cellchat_Co <- setIdent(cellchat_Co, ident.use = "CellType")

# --- CELLCHAT INFERENCE FOR EACH CONDITION ---
# Mo
cellchat_Mo <- subsetData(cellchat_Mo, features = rownames(cellchat_Mo@data))
cellchat_Mo <- identifyOverExpressedGenes(cellchat_Mo)
cellchat_Mo <- identifyOverExpressedInteractions(cellchat_Mo)
cellchat_Mo <- computeCommunProb(cellchat_Mo, type = 'truncatedMean' , trim = 0.01)
cellchat_Mo <- computeCommunProbPathway(cellchat_Mo, thresh = 0.05)
cellchat_Mo <- aggregateNet(cellchat_Mo)

# Co
cellchat_Co <- subsetData(cellchat_Co,  features = rownames(cellchat_Co@data))
cellchat_Co <- identifyOverExpressedGenes(cellchat_Co)
cellchat_Co <- identifyOverExpressedInteractions(cellchat_Co)
cellchat_Co <- computeCommunProb(cellchat_Co, type = 'truncatedMean', trim = 0.01)
cellchat_Co <- computeCommunProbPathway(cellchat_Co, thresh = 0.05)
cellchat_Co <- aggregateNet(cellchat_Co)

# Merge objects for comparison
cellchat_merged <- mergeCellChat(list(Mo = cellchat_Mo, Co = cellchat_Co), add.names = c("Mo", "Co"))

# --- DIFFERENTIAL GENE EXPRESSION AND MAPPING ---
pos.dataset <- "Co"
features.name <- "differential_genes"

# Run differential gene expression analysis
cellchat_merged <- identifyOverExpressedGenes(
  cellchat_merged,
  group.dataset = "datasets",
  pos.dataset = pos.dataset,
  features.name = features.name,
  only.pos = FALSE,
  thresh.pc = 0.1,
  thresh.fc = 0.05,
  thresh.p = 0.05,
  group.DE.combined = FALSE
)

# Map the results to the communication networks
net <- netMappingDEG(cellchat_merged, features.name = features.name, variable.all = TRUE)
write.csv(net, "net_Co_mapping_DEG_results.csv", row.names = FALSE)

net.up <- subsetCommunication(cellchat_merged, net = net, datasets = "Co", 
                                ligand.logFC = 0.01, receptor.logFC = NULL,
                                thresh = 0.05, ligand.pvalues = 0.05 )

