# DINO_3_part3_model_and_training_setup.ipynb

## Purpose
This file is part **3 of 5** from the original `DINO.ipynb`.

## What this split contains
This notebook contains a consecutive chunk of cells from the original notebook. It is intended for easier ownership by one group member while preserving the original execution order.

## Suggested ownership
Member 3

## Notes
- This split preserves the original cell order.
- Some variables, imports, outputs, or saved artifacts may depend on earlier parts.
- For final submission, validated changes should be merged back into the original notebook unless your workflow explicitly supports modular execution.

## Quick content preview
train_ds = MRNetDataset(train_df, transforms=train_tfm) valid_ds = MRNetDataset(valid_df, transforms=valid_tfm) train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True) valid_loader = DataLoader(valid_ds, batch_size...
