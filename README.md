## ISESL-SQL

The source code of paper "Semantic Enhanced Text-to-SQL Parsing via Iteratively Learning Schema Linking Graph" published at KDD 2022.

### download data

spider: https://yale-lily.github.io/spider

put the data into the data/ directory

### preprocess data

```
python3 -u preprocess/process_dataset.py --dataset_path data/train.json --raw_table_path data/tables.json --table_path data/tables.bin --output_path 'data/train.bin' --skip_large --semantic_graph
python3 -u preprocess/process_dataset.py --dataset_path data/dev.json --table_path data/tables.bin --output_path 'data/dev.bin' --skip_large --semantic_graph
python3 -u preprocess/process_graphs.py --dataset_path 'data/train.bin' --table_path data/tables.bin --output_path  data/train.rgatsql.bin
python3 -u preprocess/process_graphs.py --dataset_path 'data/dev.bin' --table_path data/tables.bin --output_path  data/dev.rgatsql.bin
```

### train model

```
CUDA_VISIBLE_DEVICES=0 python scripts/text2sql.py --task lgesql_large --seed 999 --device 0 --plm google/electra-large-discriminator --gnn_hidden_size 512 --dropout 0.2 --attn_drop 0.0 --att_vec_size 512 --model rgatsql --output_model without_pruning --score_function affine --relation_share_heads --subword_aggregation attentive-pooling --schema_aggregation head+tail --gnn_num_layers 8 --num_heads 8 --lstm onlstm --chunk_size 8 --drop_connect 0.2 --lstm_hidden_size 512 --lstm_num_layers 1 --action_embed_size 128 --field_embed_size 64 --type_embed_size 64 --no_context_feeding --batch_size 35 --grad_accumulate 5 --lr 1e-4 --l2 0.1 --warmup_ratio 0.1 --lr_schedule linear --eval_after_epoch 120 --smoothing 0.15 --layerwise_decay 0.8 --max_epoch 200 --max_norm 5 --beam_size 5 --logdir logdir/run --train_path train --dev_path dev --training --filter_edge --optimize_graph --schema_loss 
```

### Reference

```
@inproceedings{liu2022semantic,
  title={Semantic Enhanced Text-to-SQL Parsing via Iteratively Learning Schema Linking Graph},
  author={Liu, Aiwei and Hu, Xuming and Lin, Li and Wen, Lijie},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1021--1030},
  year={2022}
}
```





