# scmultisim_benchmark.R  — GRN gradient edition
# ──────────────────────────────────────────────────────────────────────────────
# Simulates a SINGLE cell population at a given GRN regulatory strength.
# Called 4x from Python (one call per gradient level).
#
# Usage:
#   Rscript scmultisim_benchmark.R <outdir> <effect> <seed>
#     outdir  — base output folder
#     effect  — GRN edge coupling strength [0.001 ... 1.0] (cascade coupling)
#     seed    — integer random seed (default 42)
#
# Saves to: <outdir>/grn_<effect_tag>/
#   matrix.mtx, features.tsv, barcodes.tsv, metadata.tsv
# ──────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(scMultiSim)
  library(Matrix)
  library(ape)
})

args       <- commandArgs(trailingOnly = TRUE)
out_dir    <- if (length(args) >= 1) args[1] else "scmultisim_simulation"
effect_val <- if (length(args) >= 2) as.numeric(args[2]) else 1.0
base_seed  <- if (length(args) >= 3) as.integer(args[3]) else 42L

# Subfolder tag: effect=0.34 -> "034"
effect_tag <- sprintf("%03.0f", effect_val * 100)
sub_dir    <- file.path(out_dir, paste0("grn_", effect_tag))
dir.create(sub_dir, showWarnings = FALSE, recursive = TRUE)
cat(sprintf("scMultiSim GRN gradient — effect=%.2f -> %s\n", effect_val, sub_dir))

set.seed(base_seed)

N_CELLS <- 500L
N_HKG   <- 50L
MU_HKG  <- 80.
R_HKG   <-  6.

# 5-gene cascade (1-indexed): G1->G2->G3->G4, G5 independent
grn <- data.frame(
  regulator = c(1L, 2L, 3L),
  target    = c(2L, 3L, 4L),
  effect    = c(effect_val, effect_val, effect_val)
)

make_gene_names <- function(n) paste0("G", seq_len(n) - 1L)

# Two near-identical pseudo-populations (required by scMultiSim internals;
# diff.cif.fraction=0.10 is the minimum safe value — fewer than 2 diff CIFs
# causes an internal dimension crash). Cells are effectively homogeneous.
tree <- read.tree(text = "((R1:1,R2:1):1);")

# Design rationale:
#   cif.mean = 1.5  LOW baseline so GRN effect dominates expression changes.
#   Previous cif.mean=4.0 gave ~100-count CIF baseline that dwarfed the GRN
#   signal — G1 log1p only varied 4.50→4.65 across all levels (0.15 units).
#   At cif.mean=1.5, baseline drops ~12x so the GRN effect is the primary driver.
#   cif.sigma = 1.0  increased cell-to-cell CIF variability for better GENIE3
#   feature importance variance (analogous to NB overdispersion R_VEC).
opts <- list(
  GRN               = grn,
  num.cells         = N_CELLS,
  num.cifs          = 20L,
  tree              = tree,
  discrete.cif      = TRUE,
  discrete.pop.size = c(as.integer(N_CELLS / 2), as.integer(N_CELLS / 2)),
  diff.cif.fraction = 0.10,
  cif.mean          = 1.5,   # LOW: keeps GRN effect as primary driver
  cif.sigma         = 1.0,   # increased variance for better GENIE3 detection
  scale.s           = 1.0,
  rand.seed         = base_seed
)

cat("  Running sim_true_counts...\n")
results <- tryCatch(
  sim_true_counts(opts),
  error = function(e) {
    cat("ERROR in sim_true_counts:", conditionMessage(e), "\n")
    quit(status = 1)
  }
)

counts_gxc <- results$counts
n_actual   <- ncol(counts_gxc)
grn_names  <- make_gene_names(nrow(counts_gxc))

# Append 50 housekeeping genes
set.seed(base_seed + 200L)
hkg_mat <- matrix(
  rnbinom(n_actual * N_HKG, size = R_HKG, mu = MU_HKG),
  nrow = N_HKG, ncol = n_actual
)
rownames(hkg_mat) <- paste0("HKG_", seq_len(N_HKG) - 1L)

full_counts <- rbind(counts_gxc, hkg_mat)
gene_names  <- c(grn_names, rownames(hkg_mat))
barcodes    <- sprintf("cell_%04d", seq_len(n_actual))
colnames(full_counts) <- barcodes

writeMM(Matrix(full_counts, sparse = TRUE), file.path(sub_dir, "matrix.mtx"))
writeLines(gene_names, file.path(sub_dir, "features.tsv"))
writeLines(barcodes,   file.path(sub_dir, "barcodes.tsv"))

meta <- data.frame(
  barcode      = barcodes,
  grn_strength = sprintf("effect_%.2f", effect_val),
  effect       = effect_val,
  pop          = results$cell_meta$pop
)
write.table(meta, file.path(sub_dir, "metadata.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)

g0_mean <- mean(counts_gxc[1, ])
g1_mean <- if (nrow(counts_gxc) >= 2) mean(counts_gxc[2, ]) else NA
cat(sprintf("  Saved: %d genes x %d cells -> %s\n",
            nrow(full_counts), n_actual, sub_dir))
cat(sprintf("  G0 mean=%.1f  G1 mean=%.1f\n", g0_mean, g1_mean))
cat("Done.\n")
