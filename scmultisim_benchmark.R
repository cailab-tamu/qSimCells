# scmultisim_benchmark.R
# ──────────────────────────────────────────────────────────────────────────────
# Simulate single-cell RNA-seq data with the official scMultiSim package
# (ZhangLabGT/scMultiSim; Zhang et al. 2023) and save outputs in
# 10x Genomics sparse format (.mtx + features.tsv + barcodes.tsv)
# for loading in Python via scipy.io.mmread.
#
# Install scMultiSim once (inside R):
#   install.packages("remotes")
#   remotes::install_github("ZhangLabGT/scMultiSim")
#
# Usage:
#   Rscript scmultisim_benchmark.R          # writes to ./scmultisim_simulation/
#   Rscript scmultisim_benchmark.R <outdir> # writes to <outdir>/
#
# Output folder structure:
#   scmultisim_simulation/
#   ├── co_culture/
#   │   ├── matrix.mtx      sparse count matrix  (genes × cells, Market Exchange)
#   │   ├── features.tsv    gene names, one per line
#   │   ├── barcodes.tsv    cell barcodes, one per line
#   │   └── metadata.tsv    cell metadata (cell_type, cluster, ...)
#   └── mono_culture/
#       ├── matrix.mtx
#       ├── features.tsv
#       ├── barcodes.tsv
#       └── metadata.tsv
#
# Read in Python:
#   import scipy.io, pandas as pd, numpy as np
#   mat      = scipy.io.mmread("co_culture/matrix.mtx").T.toarray()  # cells × genes
#   features = pd.read_csv("co_culture/features.tsv", header=None)[0].tolist()
#   barcodes = pd.read_csv("co_culture/barcodes.tsv", header=None)[0].tolist()
#   meta     = pd.read_csv("co_culture/metadata.tsv", sep="\t")
# ──────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(scMultiSim)
  library(Matrix)
  library(ape)
})

# Optional: UMAP plotting via Seurat
PLOT_UMAPS <- requireNamespace("Seurat", quietly = TRUE)
if (PLOT_UMAPS) {
  suppressPackageStartupMessages({
    library(Seurat)
    library(ggplot2)
  })
  cat("Seurat found — UMAPs will be saved.\n")
} else {
  cat("Note: install Seurat to enable UMAP plots.\n")
}

# ── Output directory ──────────────────────────────────────────────────────────
args    <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "scmultisim_simulation"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
cat(sprintf("Writing outputs to: %s\n", out_dir))

BENCHMARK_SEED <- 42L
set.seed(BENCHMARK_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Shared GRN definition
# ─────────────────────────────────────────────────────────────────────────────
# 5-gene cascade matching qSimCells and SERGIO (1-indexed, R convention):
#   G1 (master regulator) → G2 → G3 → G4    cascade
#   G5                                        independent control gene
grn <- data.frame(
  regulator = c(1L, 2L, 3L),
  target    = c(2L, 3L, 4L),
  effect    = c(1.0, 1.0, 1.0)
)

# Gene name helper: scMultiSim may output more genes than just the GRN genes
# (it adds unregulated genes via unregulated.gene.ratio, default 0.1).
# We rename all output genes sequentially as G0, G1, G2, ... so names are
# consistent and 0-indexed to match Python / SERGIO / qSimCells conventions.
make_gene_names <- function(n) paste0("G", seq_len(n) - 1L)

# ── Housekeeping gene constants (matches qSim_cell_benchmarks.ipynb) ─────────
N_HKG  <- 50L
MU_HKG <- 80.
R_HKG  <-  6.

# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper: save one simulation result as 10x sparse format
#    50 HKGs (NB mu=80, r=6) are appended so the saved data matches the
#    convention used in qSim_cell_benchmarks.ipynb and the Python benchmark.
# ─────────────────────────────────────────────────────────────────────────────
save_10x <- function(results, subfolder) {
  dir.create(subfolder, showWarnings = FALSE, recursive = TRUE)

  counts_gxc <- results$counts      # genes × cells
  meta       <- results$cell_meta

  n_cells    <- ncol(counts_gxc)
  grn_names  <- make_gene_names(nrow(counts_gxc))

  # Append 50 housekeeping genes — NB(size=R_HKG, mu=MU_HKG), all cells active
  set.seed(BENCHMARK_SEED + 200L)
  hkg_mat <- matrix(
    rnbinom(n_cells * N_HKG, size = R_HKG, mu = MU_HKG),
    nrow = N_HKG, ncol = n_cells
  )
  rownames(hkg_mat) <- paste0("HKG_", seq_len(N_HKG) - 1L)

  full_counts <- rbind(counts_gxc, hkg_mat)              # (GRN genes + 50) × cells
  gene_names  <- c(grn_names, rownames(hkg_mat))

  # Barcodes
  barcodes <- sprintf("cell_%04d", seq_len(n_cells))
  colnames(full_counts) <- barcodes

  # Write sparse matrix (genes × cells, Market Exchange format)
  writeMM(Matrix(full_counts, sparse = TRUE),
          file.path(subfolder, "matrix.mtx"))

  # Write features
  writeLines(gene_names, file.path(subfolder, "features.tsv"))

  # Write barcodes (one barcode per line)
  writeLines(barcodes, file.path(subfolder, "barcodes.tsv"))

  # Write metadata
  meta$barcode <- barcodes
  write.table(meta, file.path(subfolder, "metadata.tsv"),
              sep = "\t", row.names = FALSE, quote = FALSE)

  cat(sprintf("  Saved: %d genes (%d GRN + %d HKG) × %d cells → %s\n",
              nrow(full_counts), nrow(counts_gxc), N_HKG, n_cells, subfolder))

  invisible(list(counts = full_counts, meta = meta, gene_names = gene_names))
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Co-culture simulation  (TypeA + TypeB, 500 cells each = 1000 total)
# ─────────────────────────────────────────────────────────────────────────────
cat("Simulating co-culture...\n")
set.seed(BENCHMARK_SEED)

# discrete.cif = TRUE assigns cells to named leaf populations (discrete types).
# Without this, scMultiSim samples cells continuously along branches via
# SampleEdge(), which fails when the root branch-length is 0 or the tree
# has too few cells to distribute.  Root branch-length set to 1 (non-zero).
# discrete.pop.size gives exact cell counts per leaf (order = tree leaf order).
# sigma.b is not a valid scMultiSim parameter — removed.
tree_co <- read.tree(text = "((TypeA:1,TypeB:1):1);")

opts_co <- list(
  GRN               = grn,
  num.cells         = 1000L,
  num.cifs          = 20L,
  tree              = tree_co,
  discrete.cif      = TRUE,
  discrete.pop.size = c(500L, 500L),   # 500 TypeA + 500 TypeB
  diff.cif.fraction = 0.8,             # 80% of CIFs differ between types
  scale.s           = 1.0,
  rand.seed         = BENCHMARK_SEED
)

results_co <- sim_true_counts(opts_co)
saved_co   <- save_10x(results_co, file.path(out_dir, "co_culture"))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Mono-culture simulation  (500 cells, single homogeneous population)
# ─────────────────────────────────────────────────────────────────────────────
cat("Simulating mono-culture...\n")
set.seed(BENCHMARK_SEED + 1L)

# scMultiSim v1.2 requires ≥ 2 leaf nodes; single-leaf trees crash internally.
# Fix: two-leaf tree with diff.cif.fraction = 0.10, so only 2/20 CIFs differ
# between the two "populations."  They are effectively the same cell type —
# a genuine mono-culture whose UMAP forms a single cluster.  This is
# fundamentally different from co-culture (diff.cif.fraction = 0.8), where
# the two populations are clearly separated.
# (fraction < 0.10 gives <2 differential CIFs, which collapses the internal
# DE matrix to a vector and crashes scMultiSim with "incorrect number of
# dimensions")
tree_mono <- read.tree(text = "((R1:1,R2:1):1);")
opts_mono <- list(
  GRN               = grn,
  num.cells         = 500L,
  num.cifs          = 20L,
  tree              = tree_mono,
  discrete.cif      = TRUE,
  discrete.pop.size = c(250L, 250L),  # two near-identical replicates
  diff.cif.fraction = 0.10,           # 2/20 CIFs differ → single cluster
  # NOTE: fraction < 0.10 (i.e. < 2 diff CIFs) collapses the internal DE
  # matrix to a vector and crashes with "incorrect number of dimensions"
  scale.s           = 1.0,
  rand.seed         = BENCHMARK_SEED + 1L
)
results_mono <- sim_true_counts(opts_mono)
saved_mono   <- save_10x(results_mono, file.path(out_dir, "mono_culture"))
cat("  Saved mono_culture:", ncol(results_mono$counts), "cells\n")

# ─────────────────────────────────────────────────────────────────────────────
# 5. UMAP visualisation
# ─────────────────────────────────────────────────────────────────────────────
plot_umap_seurat <- function(saved, title_prefix) {
  # saved$counts already contains GRN genes + 50 HKGs (added in save_10x)
  all_counts <- saved$counts    # (GRN + HKG) × cells
  meta       <- saved$meta
  all_genes  <- saved$gene_names
  n_cells    <- ncol(all_counts)

  # Cell-type label from scMultiSim 'pop' column (1 = TypeA, 2 = TypeB)
  cell_type <- if ("pop" %in% colnames(meta))
    paste0("Type", LETTERS[meta$pop]) else rep("TypeA", n_cells)

  rownames(all_counts) <- all_genes
  colnames(all_counts) <- sprintf("cell_%04d", seq_len(n_cells))

  # Build Seurat object — skip FindVariableFeatures (small gene set)
  seu <- CreateSeuratObject(counts = all_counts, project = title_prefix)
  seu$cell_type <- cell_type
  Idents(seu)   <- "cell_type"
  seu <- NormalizeData(seu, verbose = FALSE)
  seu <- ScaleData(seu, features = rownames(seu), verbose = FALSE)

  n_pc <- min(10L, nrow(all_counts) - 1L)
  seu  <- RunPCA(seu,  features = rownames(seu), npcs = n_pc,
                 verbose = FALSE, seed.use = BENCHMARK_SEED)
  seu  <- RunUMAP(seu, dims = seq_len(n_pc),
                  verbose = FALSE, seed.use = BENCHMARK_SEED)

  # Two panels: cell type + G0 (cascade driver) — display inline, no file save
  p1 <- DimPlot(seu, reduction = "umap", group.by = "cell_type",
                pt.size = 0.8, label = TRUE) +
        ggtitle(paste(title_prefix, "— cell type"))
  p2 <- FeaturePlot(seu, features = all_genes[1], reduction = "umap",
                    pt.size = 0.8, cols = c("grey90", "#C0392B")) +
        ggtitle(paste(title_prefix, "—", all_genes[1]))

  print(p1 + p2)
}

if (PLOT_UMAPS) {
  plot_umap_seurat(saved_co,   "scMultiSim co-culture")
  plot_umap_seurat(saved_mono, "scMultiSim mono-culture")
} else {
  cat("Skipping UMAPs (Seurat not installed).\n")
}

cat("Done.\n")
