# DINO_5_part5_evaluation_and_analysis.ipynb

## Purpose
This file is part **5 of 5** from the original `DINO.ipynb`.

## What this split contains
This notebook contains a consecutive chunk of cells from the original notebook. It is intended for easier ownership by one group member while preserving the original execution order.

## Suggested ownership
Member 5

## Notes
- This split preserves the original cell order.
- Some variables, imports, outputs, or saved artifacts may depend on earlier parts.
- For final submission, validated changes should be merged back into the original notebook unless your workflow explicitly supports modular execution.

## Quick content preview
rows = [] for i, name in enumerate(CFG.LABEL_NAMES): tn, fp, fn, tp = confusion_matrix(y_true[:,i], y_preds[:,i]).ravel() rows.append({ "Condition" : name.upper(), "Accuracy" : accuracy_score(y_true[:,i], y_preds[:,i]), "Sensitivity": tp/(tp+fn) if (tp+fn)>0 else 0, "Specificity": tn/(tn+fp) if (tn+...
