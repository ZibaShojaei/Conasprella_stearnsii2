##10-26
##Exclude 2009-only project data
##removed correlated varaibles Greg said
##oversample the entire dataset first, create the new balanced dataframe, then do the train/test split and CV.
## As drew said
#exclude lat and long


# ========= PACKAGES =========

# === Load main R packages for data prep, modeling, and evaluation ===
# readr, dplyr, tidyr → data loading and cleaning
# recipes → preprocessing steps for modeling
# workflows → combine recipe + model into one workflow
# parsnip → define and train models (like XGBoost)
# dials, tune → parameter tuning and grid search
# yardstick → performance metrics (AUC, accuracy, etc.)
# rsample → data splitting and cross-validation
# themis → handle class imbalance (oversampling)
# parallel → speed up tuning with multiple cores
# tidymodels → loads the full modeling framework conveniently


library(readr)
library(dplyr)
library(tidyr)
library(recipes)
library(workflows)
library(parsnip)
library(dials)
library(tune)
library(yardstick)
library(rsample)
library(themis)       
library(parallel)
library(tidymodels)
library(sf)

set.seed(1876)

# ========= 0) LOAD & LABEL =========
DATA <- readr::read_csv(
  "Conasprella_stearnsii_1995-2023_oysterD_no2009.csv",
  na = c("NULL", "", "NA")
)

## Makes a normal data frame and adds a row ID called "source_id".
### "source_id" gives each record a unique ID — used later to group or split data safely.

DATA <- DATA %>%
  select(-any_of("source_id")) %>%
  tibble::rowid_to_column("source_id")

# Exclude 2009-only project data
DATA <- DATA %>% filter(as.numeric(Year) != 2009)

readr::write_csv(DATA, "Conasprella_stearnsii_1995-2023_oysterD_no2009.csv")


# ========= 1) DROP LEAKY TARGET-DERIVED FIELDS =========
DF_BASE <- DATA %>%
  dplyr::select(-dplyr::any_of(c("Conasprella.stearnsii","Year","Latitude","Longitude")))

# ========= 1) CREATE PRESENCE/ABSENCE COLUMN FIRST =========
# Make a copy of DATA with presence/absence defined 
DF_BASE <- DATA %>%
  mutate(cone_pa = ifelse(Conasprella.stearnsii > 0, 1, 0)) %>%
  dplyr::select(-dplyr::any_of(c("Conasprella.stearnsii","Year","Latitude","Longitude")))

# ========= 2) OVERSAMPLE FIRST =========
presence_idx <- which(DF_BASE$cone_pa == 1)
pos_n <- sum(DF_BASE$cone_pa == 1); neg_n <- sum(DF_BASE$cone_pa == 0)
add_n <- max(neg_n - pos_n, 0)
oversample_idx <- if (add_n > 0) sample(presence_idx, size = add_n, replace = TRUE) else integer(0)
DF_OS <- dplyr::bind_rows(DF_BASE, DF_BASE %>% dplyr::slice(oversample_idx))

# keep G1_SAV1 as single numeric variable
DF_OS$G1_SAV1 <- as.numeric(as.factor(DF_OS$G1_SAV1))

cat("Class balance after oversampling:\n"); print(table(DF_OS$cone_pa))

# --- 3) You split the balanced dataset into training (75%) and testing (25%) sets
stopifnot("source_id" %in% names(DF_OS))
DF_OS$cone_pa <- factor(DF_OS$cone_pa, levels = c(1,0))
SPL <- rsample::group_initial_split(DF_OS, prop = 0.75, group = source_id, strata = cone_pa)
TRN <- training(SPL); TST <- testing(SPL)



# --- 4) Create preprocessing recipe for model training ---
# Defines how predictors are cleaned and prepared before modeling.

rec_cls <- recipe(cone_pa ~ ., data = TRN) %>%
  step_rm(any_of(c("source_id", "Year", "Latitude", "Longitude", "Conasprella.stearnsii"))) %>%   # remove ID and non-predictor columns
  step_zv(all_predictors()) %>%                                # drop predictors with zero variance
  step_impute_median(all_numeric_predictors()) %>%              # fill missing numeric values with median
  step_string2factor(all_nominal_predictors()) %>%              # convert text columns to factors
  step_novel(all_nominal_predictors()) %>%                      # handle unseen categories in new data
  step_unknown(all_nominal_predictors())                        # handle missing factor levels as "unknown"
# (No step_dummy) → keeps G1_SAV1 as a single numeric-like factor variable



# Apply the recipe to clean the training data and make a predictor table for tuning.

prep_tmp <- prep(rec_cls)
pred_mat <- bake(prep_tmp, new_data = NULL) %>% dplyr::select(-cone_pa)


# Define XGBoost model and create 8 random parameter sets for tuning

xgb_spec_cls <- parsnip::boost_tree(
  trees = tune(), tree_depth = tune(), min_n = tune(),
  loss_reduction = tune(), mtry = tune(),
  sample_size = 0.8, learn_rate = 0.10
) %>%
  parsnip::set_engine("xgboost", eval_metric = "logloss", nthread = max(1, parallel::detectCores()-1)) %>%
  parsnip::set_mode("classification")

xgb_grid_cls <- dials::grid_latin_hypercube(
  dials::trees(range = c(200, 600)),       # number of trees
  dials::tree_depth(range = c(2, 6)),      # depth of each tree
  dials::min_n(),                          # min obs per leaf
  dials::loss_reduction(),                 # gamma regularization
  dials::finalize(dials::mtry(), pred_mat),# number of variables tried at each split
  size = 8                                 # total combinations to test
)


# --- 6) Train and tune the XGBoost model ---
#Choose the best tuning values (highest ROC-AUC) and use them to build the final model


wf_cls <- workflow() %>% add_recipe(rec_cls) %>% add_model(xgb_spec_cls)
folds <- rsample::group_vfold_cv(TRN, group = source_id, v = 3, strata = cone_pa)
tune_res <- tune::tune_grid(
  wf_cls, resamples = folds, grid = xgb_grid_cls,
  control = tune::control_grid(save_pred = FALSE, verbose = TRUE),
  metrics = yardstick::metric_set(yardstick::roc_auc, yardstick::pr_auc, yardstick::accuracy)
)
best_params <- tune::select_best(tune_res, metric = "roc_auc"); print(best_params)


# --- 7) Fit the final XGBoost model and make predictions ---
# Train the model using the best tuning parameters,
# then predict probabilities and classes on the test set
# and combine them into one evaluation table.

final_wf <- tune::finalize_workflow(wf_cls, best_params)
fit_cls  <- parsnip::fit(final_wf, data = TRN)
pred_prob <- predict(fit_cls, TST, type = "prob")
pred_cls  <- predict(fit_cls, TST, type = "class")
eval_tbl  <- dplyr::bind_cols(TST %>% dplyr::select(cone_pa), pred_prob, pred_cls)

# --- 7) Final metrics + confusion matrix ---
cat("\n=== TEST METRICS (classification) ===\n")

# ROC-AUC
print(yardstick::roc_auc(eval_tbl, truth = cone_pa, .pred_1, event_level = "first"))

# PR-AUC
print(yardstick::pr_auc(eval_tbl, truth = cone_pa, .pred_1, event_level = "first"))

# Accuracy
print(yardstick::accuracy(eval_tbl, truth = cone_pa, .pred_class))

# Confusion matrix
print(yardstick::conf_mat(eval_tbl, truth = cone_pa, estimate = .pred_class))


###############results 
#est ROC-AUC: 0.834
#Test PR-AUC: 0.774
#Test Accuracy: 0.699

##Confusion Matrix
#  TP: 255 TN: 525 FP: 305 FN: 31



# --- 7b) Confusion matrix heatmap (no yardstick::autoplot) ---
# Compute confusion matrix from evaluation table
cm_cls <- yardstick::conf_mat(eval_tbl, truth = cone_pa, estimate = .pred_class)

# Convert confusion matrix to data frame for plotting
cm_df <- as.data.frame(cm_cls$table); names(cm_df) <- c("truth","prediction","n")

# Plot confusion matrix as heatmap with counts
ggplot2::ggplot(cm_df, ggplot2::aes(x = prediction, y = truth, fill = n)) +
  ggplot2::geom_tile() +
  ggplot2::geom_text(ggplot2::aes(label = n)) +
  ggplot2::labs(title = "Confusion Matrix — Test Set", 
                x = "Predicted", y = "Truth", fill = "Count") +
  ggplot2::theme_minimal()


# --- 7c) Variable importance ---
xgb_parsnip <- workflows::extract_fit_parsnip(fit_cls)
vip::vip(xgb_parsnip$fit, num_features = 20) +
  ggplot2::labs(title = "Top 20 Variable Importance — XGBoost (Classification, cleaned)")


# --- 8) Save classification artifacts ---
saveRDS(fit_cls, file = "xgb_cls_final.rds")
readr::write_csv(dplyr::bind_cols(TST, pred_prob, pred_cls), "xgb_cls_test_predictions.csv")
cat("\nSaved: xgb_cls_final.rds, xgb_cls_test_predictions.csv\n") 


################### --- R1) Build regression frame (drop leaks + coords + PA) ---#########
# Prepare data for regression model ---
# Remove presence/absence and coordinate columns,
# then make sure the abundance variable is numeric for regression.

DF_REG <- DATA %>%
  dplyr::select(-dplyr::any_of(c(
    "cone_pa","Latitude","Longitude"
  )))

DF_REG$Conasprella.stearnsii <- suppressWarnings(as.numeric(DF_REG$Conasprella.stearnsii))

# --- R2) Split regression data into training and testing sets ---
# Create a unique key (source_id + Year) to group related samples,
# then randomly select 75% of these groups for training and 25% for testing.

DF_REG <- DF_REG %>% dplyr::mutate(key = paste0(source_id, "__", Year))
train_keys <- sample(unique(DF_REG$key), size = floor(0.75 * dplyr::n_distinct(DF_REG$key)))
TRN_REG <- DF_REG %>% dplyr::filter(key %in% train_keys)
TST_REG <- DF_REG %>% dplyr::filter(!key %in% train_keys)
cat("TRN_REG rows:", nrow(TRN_REG), " TST_REG rows:", nrow(TST_REG), "\n")

# --- R3) Recipe (drop IDs; encode; impute; OHE) ---
# Remove ID and non-predictor columns, fill missing numeric values,
# convert text to factors, handle new/unknown categories,
# and one-hot encode all categorical variables for regression.


rec_reg <- recipes::recipe(Conasprella.stearnsii ~ ., data = TRN_REG) %>%
  recipes::step_rm(dplyr::any_of(c("source_id","Year","key"))) %>%  # fixed: replaced StationNumber with source_id
  recipes::step_zv(recipes::all_predictors()) %>%
  recipes::step_impute_median(recipes::all_numeric_predictors()) %>%
  recipes::step_string2factor(recipes::all_nominal_predictors()) %>%
  recipes::step_novel(recipes::all_nominal_predictors()) %>%
  recipes::step_unknown(recipes::all_nominal_predictors()) %>%
  recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = TRUE)



# --- R3b) Prepare recipe and build predictor matrix ---
# Apply all preprocessing steps and create a clean predictor dataset
# (without the target variable) for tuning the regression model.

prep_tmp_reg <- recipes::prep(rec_reg)
pred_mat_reg <- recipes::bake(prep_tmp_reg, new_data = NULL) %>% dplyr::select(-Conasprella.stearnsii)


# --- R4) Define XGBoost regression model and tuning grid ---
# Regularized XGBoost: smaller tree depth, higher min_n, and fewer trees to reduce point-like predictions


xgb_spec_reg <- parsnip::boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  mtry = tune(),
  sample_size = 0.8,
  learn_rate = 0.05   # smoother learning to reduce overfitting
) %>%
  parsnip::set_engine("xgboost",
                      eval_metric = "rmse",
                      nthread = max(1, parallel::detectCores() - 1)) %>%
  parsnip::set_mode("regression")

xgb_grid_reg <- dials::grid_latin_hypercube(
  dials::trees(range = c(200, 600)),        # fewer trees (prevent overfit)
  dials::tree_depth(range = c(2, 4)),       # shallower trees
  dials::min_n(range = c(5, 15)),           # larger node size
  dials::loss_reduction(range = c(0, 5)),   # extra regularization
  dials::finalize(dials::mtry(), pred_mat_reg),
  size = 12
)


# --- R5) Train and tune the XGBoost regression model ---
# Combine the recipe and model into a workflow,
# run 3-fold cross-validation to tune parameters,
# and select the best set based on the lowest RMSE (error).

wf_reg <- workflows::workflow() %>% workflows::add_recipe(rec_reg) %>% workflows::add_model(xgb_spec_reg)
folds_reg <- rsample::vfold_cv(TRN_REG, v = 3)
tune_res_reg <- tune::tune_grid(
  wf_reg, resamples = folds_reg, grid = xgb_grid_reg,
  control = tune::control_grid(save_pred = FALSE, verbose = TRUE),
  metrics = yardstick::metric_set(yardstick::rmse, yardstick::mae, yardstick::rsq)
)
best_params_reg <- tune::select_best(tune_res_reg, metric = "rmse"); print(best_params_reg)

# --- R6) Fit the final regression model and evaluate performance ---
# Train the model with the best parameters on training data,
# predict abundance values on the test set,
# and calculate RMSE, MAE, and R² to assess model accuracy.

final_wf_reg <- tune::finalize_workflow(wf_reg, best_params_reg)
fit_reg <- parsnip::fit(final_wf_reg, data = TRN_REG)
pred_reg <- predict(fit_reg, TST_REG) %>% dplyr::bind_cols(TST_REG %>% dplyr::select(Conasprella.stearnsii))
names(pred_reg) <- c(".pred","truth")
cat("\n=== TEST METRICS (regression) ===\n")
print(yardstick::rmse(pred_reg, truth = truth, estimate = .pred))
print(yardstick::mae( pred_reg, truth = truth, estimate = .pred))
print(yardstick::rsq( pred_reg, truth = truth, estimate = .pred))


##########regression result##########
#4_2

#RMSE: 3.14
#MAE: 0.664
#R²: 0.698



# --- R7) Save regression artifacts ---
saveRDS(fit_reg, "xgb_reg_final.rds")
readr::write_csv(dplyr::bind_cols(TST_REG, .pred = pred_reg$.pred), "xgb_reg_test_predictions.csv")
cat("\nSaved: xgb_reg_final.rds, xgb_reg_test_predictions.csv\n")



############# ROC and Precision-Recall curves for the final classification model
library(ggplot2)

# ROC curve
roc_curve(eval_tbl, truth = cone_pa, .pred_1, event_level = "first") %>%
  autoplot() + ggplot2::labs(title = "ROC Curve — XGBoost Classification")

# PR curve
pr_curve(eval_tbl, truth = cone_pa, .pred_1, event_level = "first") %>%
  autoplot() + ggplot2::labs(title = "Precision–Recall Curve — XGBoost Classification")



########## STUDY AREA (buffer around species)

library(sf)
library(dplyr)
library(smoothr)

# keep valid water points
DATA <- read_csv("Conasprella_stearnsii_1995-2023_oysterD_no2009.csv")

pts_sf <- DATA %>%
  filter(!is.na(Longitude), !is.na(Latitude), Depth.B > 0) %>%
  st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326)


# make single merged buffer polygon
study_area <- pts_sf |>
  st_transform(32617) |>
  st_buffer(20000) |>         # large enough to connect all
  st_union() |>
  st_make_valid() |>
  st_transform(4326)

# save and plot
st_write(study_area, "study_area_bay_water.shp", delete_layer = TRUE, quiet = TRUE)
plot(st_geometry(study_area), col = "lightblue", main = "Study Area")
plot(pts_sf, add = TRUE, col = "red", pch = 20)


#############base map
library(sf)
library(ggplot2)
library(ggspatial)

ggplot() +
  annotation_map_tile(type = "cartolight", zoomin = -1)+
  geom_sf(data = study_area, fill = NA, color = "red", linewidth = 1) +
  geom_sf(data = pts_sf, color = "darkred", size = 1) +
  coord_sf(expand = TRUE) +
  theme_minimal()


#####################################prediction map####################

#### STEP 2 — Build predictor raster stack by IDW interpolation

library(readr); library(dplyr); library(sf); library(terra); library(gstat); library(sp)

# --- load data + study area ---
DATA <- read_csv("Conasprella_stearnsii_1995-2023_oysterD_no2009.csv",
                 na = c("NULL","","NA")) |> data.frame()
study_area <- st_read("study_area_bay_water.shp") |> 
  st_make_valid()
saveRDS(study_area, "study_area.rds")



# drop target + coords (keep only predictors)
# drop target + coords + non-environmental fields (avoid Year leakage)
drops <- c("Conasprella.stearnsii","cone_pa","Latitude","Longitude","source_id",
           "Year","StationNumber")

preds <- DATA |> dplyr::select(-dplyr::any_of(drops))


# numeric variables only
num_vars <- names(preds)[sapply(preds, is.numeric)]

# project study area + points
sa_m  <- st_transform(study_area, 32617)   # UTM 17N, adjust EPSG if needed
pts_m <- st_as_sf(DATA, coords = c("Longitude","Latitude"), crs = 4326) |>
  st_transform(32617)

# template raster (250 m resolution)
tmpl <- rast(ext(vect(sa_m)), resolution = 100, crs = crs(vect(sa_m)))
# coarser 500 m grid to smooth predictions and reduce point-like patterns


#IDW
#helper function for IDW

idw_raster <- function(var, pts, tmpl, sa_m, DATA) {
  ok <- is.finite(DATA[[var]])
  if (sum(ok) < 3) return(NULL)
  
  # Prepare spatial points
  sp_pts <- as(pts[ok, ], "Spatial")
  sp_pts$val <- DATA[[var]][ok]
  
  # IDW interpolation (smoother settings)
  gs <- gstat::gstat(formula = val ~ 1, data = sp_pts, nmax = 40, set = list(idp = 0.5))
  
  # Create grid for interpolation
  grd_df <- as.data.frame(terra::xyFromCell(tmpl, 1:terra::ncell(tmpl)))
  names(grd_df) <- c("x", "y")
  coordinates(grd_df) <- ~x + y
  gridded(grd_df) <- TRUE
  proj4string(grd_df) <- proj4string(sp_pts)
  
  # Predict and safely convert to raster
  pred <- try(predict(gs, grd_df), silent = TRUE)
  if (inherits(pred, "try-error")) return(NULL)
  
  r_idw <- try(terra::rast(as(pred["var1.pred"], "SpatialPixelsDataFrame")), silent = TRUE)
  if (inherits(r_idw, "try-error") || is.null(r_idw) || nlyr(r_idw) == 0) return(NULL)
  
  # Mask and smooth
  names(r_idw) <- var
  r_idw <- terra::mask(r_idw, vect(sa_m))
  r_idw <- terra::focal(r_idw, w = 3, fun = mean, na.policy = "omit", na.rm = TRUE, fillvalue = NA)
  
  return(r_idw)
}


#IDW end 


# run IDW for all numeric predictors
rast_list <- lapply(num_vars, idw_raster, pts = pts_m, tmpl = tmpl, sa_m = sa_m, DATA = DATA)
rast_list <- rast_list[!sapply(rast_list, is.null)]
pred_stack <- rast(rast_list)

# save stack
writeRaster(pred_stack, "predictor_stack.tif", overwrite = TRUE)

# quick check
print(pred_stack)
names(pred_stack)
plot(pred_stack[[1]]); plot(vect(sa_m), add = TRUE, col = NA, lwd = 2)

###### STEP 3 — Convert raster stack into dataframe

# load predictor stack
pred_stack <- rast("predictor_stack.tif")

# convert raster cells to rows (x, y, predictor values)
pred_df <- as.data.frame(pred_stack, xy = TRUE, na.rm = TRUE)

# quick check
glimpse(pred_df)

# save for later steps
saveRDS(pred_df, "predictor_grid_df.rds")


############# STEP 4 (fixed) — Add categorical predictors with nearest-neighbor

# reload grid of numeric predictors
#loads the saved prediction grid with numeric predictors (and coordinates) back into R so you can add more info to it.
pred_df <- readRDS("predictor_grid_df.rds")

# categorical variables
cat_vars <- c("G1_SAV1")


## take only rows where categorical data exists
cats <- DATA[, c("Longitude","Latitude","G1_SAV1")]
cats <- cats[!is.na(cats$G1_SAV1), ]

# convert to sf
pts_cat <- st_as_sf(cats, coords = c("Longitude","Latitude"), crs = 4326) |>
  st_transform(32617)

# grid (from numeric raster dataframe)
# RIGHT (x,y are UTM Zone 17N)
grid_sf <- st_as_sf(pred_df, coords = c("x","y"), crs = 32617)

# nearest-neighbor match
nn <- st_nearest_feature(grid_sf, pts_cat)
pred_cats <- st_drop_geometry(pts_cat[nn, "G1_SAV1"])

# rebuild dataframe: numeric + categorical
pred_full <- cbind(pred_df[, c("x","y")], pred_df[, !(names(pred_df) %in% c("x","y"))], pred_cats)

# clean categorical values
pred_full$G1_SAV1[is.na(pred_full$G1_SAV1)] <- "Unknown"
pred_full$G1_SAV1 <- as.factor(pred_full$G1_SAV1)

# save and check
saveRDS(pred_full, "predictor_grid_with_cats.rds")
head(pred_full)


######## --- STEP 5: Predict on the grid using your trained XGBoost model ---

library(workflows)
library(recipes)
library(parsnip)


# 1. Keep coordinates separate (needed later for mapping)
coords <- pred_full[, c("x", "y")]

# 2. Prepare predictor dataframe
X <- pred_full[, setdiff(names(pred_full), c("x","y","geometry"))]
X$source_id <- seq_len(nrow(X))   # unique ID for each row

# ✅ Fix data types
# Convert G1_SAV1 to numeric (model expects it as numeric)
if ("G1_SAV1" %in% names(X)) {
  X$G1_SAV1 <- suppressWarnings(as.numeric(as.factor(X$G1_SAV1)))
}

# Ensure all numeric predictors are numeric
num_cols <- c("Depth.B","Temp.B","Sal.B","DO.B","pH.B",
              "SiltClayPercent","TOC","SAV1.P.A","SAV2.P.A","dist_oyster_m","G1_SAV1")
for (v in intersect(num_cols, names(X))) {
  X[[v]] <- suppressWarnings(as.numeric(X[[v]]))
}

# 3. Load your trained classification model
fit_cls <- readRDS("xgb_cls_final.rds")

# 4️⃣ Predict probabilities for each grid cell safely
# Make sure all required predictors exist
required_vars <- c("Depth.B","Temp.B","Sal.B","DO.B","pH.B",
                   "SiltClayPercent","TOC","SAV1.P.A","SAV2.P.A","dist_oyster_m","G1_SAV1")

missing_vars <- setdiff(required_vars, names(X))
for (v in missing_vars) X[[v]] <- 0  # add missing variables if needed


# ... all your previous steps creating X ...

# ✅ Insert the check + fix block HERE
if (nrow(X) == 0) stop("X is empty — prediction grid not created. Check your IDW results before proceeding.")

missing_vars <- setdiff(c("Depth.B","Temp.B","Sal.B","DO.B","pH.B",
                          "SiltClayPercent","TOC","SAV1.P.A","SAV2.P.A",
                          "dist_oyster_m","G1_SAV1"), names(X))
if (length(missing_vars) > 0) {
  for (v in missing_vars) {
    X[[v]] <- rep(0, nrow(X))
  }
}

# THEN run your prediction line
pred_probs <- predict(fit_cls, new_data = X, type = "prob")



# 5. Combine coordinates + predictions into final output
pred_out <- cbind(coords, pred_probs)

# 6. Save results
saveRDS(pred_out, "xgb_prob_grid.rds")

# 7. Quick check and save raster
head(pred_out)

# --- Save and smooth prediction raster for mapping ---
# Create raster from predicted probabilities
r_prob <- rast(pred_out[, c("x", "y", ".pred_1")], type = "xyz", crs = "EPSG:32617")

# Aggregate to coarser grid (2x2 cells) to reduce point-like patterns
r_prob <- terra::aggregate(r_prob, fact = 2, fun = mean, na.rm = TRUE)

# Save final smoothed raster for later mapping
writeRaster(r_prob, "xgb_prob_map.tif", overwrite = TRUE)


##########
library(terra); library(sf); library(ggplot2); library(ggspatial)

# ---- Correct raster creation and projection ----
pred_out <- readRDS("xgb_prob_grid.rds")

# make sure coordinates are numeric
pred_out$x <- as.numeric(pred_out$x)
pred_out$y <- as.numeric(pred_out$y)

# 1️⃣ Raster in UTM (EPSG 32617)
r_prob <- rast(pred_out[, c("x","y",".pred_1")], type = "xyz", crs = "EPSG:32617")

# 2️⃣ Reproject to WGS84 for basemap
r_prob_wgs <- terra::project(r_prob, "EPSG:4326", method = "bilinear")

# 3️⃣ Name and convert to dataframe
names(r_prob_wgs) <- "prob_presence"
r_df <- as.data.frame(r_prob_wgs, xy = TRUE, na.rm = TRUE)


# --- Final map: predicted habitat raster with coastline and county outlines ---
library(terra)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)

# 1️⃣ Load your prediction raster (already saved)
r_prob <- rast("xgb_prob_map.tif")

# 2️⃣ Reproject to WGS84 for consistent overlays
r_prob_wgs <- terra::project(r_prob, "EPSG:4326", method = "bilinear")


# 3️⃣ Load detailed Florida coastline shapefile (from Box)
fl_outline <- st_read("Detailed_Florida_State_Boundary.shp") |> 
  st_make_valid() |> 
  st_transform(4326)



# 4️⃣ Plot predicted raster + detailed Florida outline
plot(r_prob_wgs,
     main = "Predicted Habitat Suitability — XGBoost Model",
     col = colorRampPalette(c("gray60","lightgreen","yellow","orange","red"))(100),
     colNA = "gray90",
     range = c(0,1))





# ===########## Binary habitat map (suitable / unsuitable) ===
library(terra)
library(sf)
library(dplyr)
library(ggplot2)
library(rnaturalearth)
library(rnaturalearthdata)
library(pROC)   # <-- this is the missing package (for roc())

# 1️⃣ Load probability raster
r_prob <- rast("xgb_prob_map.tif")

# 2️⃣ Load study area
sa <- readRDS("study_area.rds") |>
  st_make_valid() |> st_transform(4326)

# 3️⃣ Reload observed presence data
DATA <- read.csv("Conasprella_stearnsii_1995-2023_oysterD_no2009.csv",
                 na.strings = c("NA","NULL",""))
DATA$presence <- ifelse(DATA$Conasprella.stearnsii > 0, 1, 0)

# Convert observation points to sf and match raster CRS
pts <- st_as_sf(DATA, coords = c("Longitude", "Latitude"), crs = 4326) |>
  st_transform(crs(r_prob))

# Extract predicted probabilities at presence/absence locations
DATA$pred_prob <- terra::extract(r_prob, vect(pts))[, 2]

# 4️⃣ Find best threshold (max sensitivity + specificity)
roc_obj <- roc(DATA$presence, DATA$pred_prob)
thr <- as.numeric(coords(roc_obj, "best", ret = "threshold"))
thr  # print threshold value (example ≈ 0.45)

# 5️⃣ Create binary raster (0 = unsuitable, 1 = suitable)
r_bin <- classify(r_prob, rbind(c(-Inf, thr, 0),
                                c(thr, Inf, 1)))

# 6️⃣ Convert to WGS84 for plotting
r_bin_wgs <- project(r_bin, "EPSG:4326")
r_df <- as.data.frame(r_bin_wgs, xy = TRUE, na.rm = TRUE)
names(r_df)[3] <- "suitability"


# 7️⃣ Load detailed Florida coastline shapefile (better resolution)
fl_outline <- st_read("Detailed_Florida_State_Boundary.shp") |>
  st_make_valid() |> 
  st_transform(4326)


# 8️⃣ Plot final binary habitat map

# --- Final clean binary map with fixed legend size and style ---

par(mar = c(4, 4, 4, 6))  # space for legend on the right

# Plot binary raster
plot(r_bin_wgs,
     col = c("gray80", "purple"),
     legend = FALSE,
     main = "",
     axes = TRUE,
     box = TRUE)

# Add outlines
plot(st_geometry(fl_outline), add = TRUE, border = "gray40", lwd = 0.8)
plot(st_geometry(sa), add = TRUE, border = "red", lwd = 0.8)

# Add title
mtext("Binary Habitat Map — XGBoost Model", side = 3, line = 1, adj = 0, font = 2, cex = 1.1)

# Add properly sized legend
par(xpd = TRUE)
legend("topright",
       inset = c(-0.05, 0),
       legend = c("Unsuitable habitat", "Suitable habitat"),
       fill = c("gray80", "purple"),
       border = "black",
       bg = "white",
       box.lwd = 0.5,
       cex = 0.8,     # balanced text size
       pt.cex = 0.9)  # box size slightly smaller
par(xpd = FALSE)




