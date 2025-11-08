python train_qformer.py \
    --pretrained_model_path ./data/ddpm_dit_cifar_100_epochs.pth \
    --dense_captions_path "./data/cifar10_dense_captions.jsonl" \
    --epochs 50 \
    --batch_size 128 --lr 1e-4 \
    --save_model_path /ocean/projects/cis220031p/ruilobap/data/trained_qformer.pth \
    --gpt2_layer_index 12 --num_query_tokens 4 --device 0 --cfg 3.0 --data_dir ./data \
    --gpt2_cache_dir ./data --optimizer_ckpt_interval 5 \
    --cache_text_embeddings ./data/text_embeddings.pt