python3 train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/lstm \
    --batch_size=1000 \
    --num_epochs=1 \
    --val_every=5 \
    --embedding_dim=31 \
    --force_cpu 