CONFIGS="conf/tuning/sti_transformer_lr10.0_ag8_p.5_mlm.yaml conf/tuning/sti_transformer_lr10.0_ag8_p.5_no_kl.yaml conf/tuning/sti_transformer_lr10.0_ag8_p.5_teachnomask.yaml conf/tuning/sti_transformer_lr10.0_ag8_p.5_mse.yaml"

for config in $CONFIGS; do ./encode.sh dump/fluent/deltafalse/data.fluent.json exp/train_960_pytorch_$(basename ${config%.*})_specaug; done
