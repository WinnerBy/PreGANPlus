# Ablation Studies Checkpoint Directory

This directory stores checkpoint files for the four ablation studies:

1. **AblationNoTransformer**: FPE_16 encoder (GRU+GAT, same as PreGAN) instead of Transformer
   - Files: `simulator_FPE_16.ckpt`, `simulator_Gen_16_MigrationAware_ablation_notrans.ckpt`, `simulator_Disc_16_MultiObjective_ablation_notrans.ckpt`

2. **AblationNoGAT**: Transformer encoder without GAT module
   - Files: `simulator_TransformerNoGAT_16.ckpt`, `simulator_Gen_16_MigrationAware_ablation_nogat.ckpt`, `simulator_Disc_16_MultiObjective_ablation_nogat.ckpt`

3. **AblationNoMigrationAware**: Transformer encoder with standard generator (no migration awareness)
   - Files: `simulator_Transformer_16.ckpt`, `simulator_Gen_16_ablation_nomigaware.ckpt`, `simulator_Disc_16_MultiObjective_ablation_nomigaware.ckpt`

4. **AblationNoMultiObjective**: Transformer encoder with migration-aware generator but standard discriminator
   - Files: `simulator_Transformer_16.ckpt`, `simulator_Gen_16_MigrationAware_ablation_nomulti.ckpt`, `simulator_Disc_16_ablation_nomulti.ckpt`

## Training

To train encoders only (without GAN), use the `encoder_only=True` parameter:

```python
from recovery import AblationNoTransformerRecovery

ablation = AblationNoTransformerRecovery(hosts=16, env='simulator', training=True, encoder_only=True)
# This will only train the encoder without loading/training GAN models
```

## Directory Separation

All ablation checkpoints are stored separately from main model checkpoints to:
- Avoid conflicts with main PreGAN, PreGANPlus, PreGANPlusEnhanced models
- Make it easy to isolate and compare ablation study results
- Facilitate cleanup and archiving of experiment variants
