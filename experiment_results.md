root@4b0009fd93b32486:~/automated-ai-research-experiments# git pull
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Compressing objects: 100% (1/1), done.
remote: Total 3 (delta 2), reused 3 (delta 2), pack-reused 0 (from 0)
Unpacking objects: 100% (3/3), 260 bytes | 260.00 KiB/s, done.
From https://github.com/vukrosic/automated-ai-research-experiments
   afc0ad6..4373e69  main       -> origin/main
Updating afc0ad6..4373e69
Fast-forward
 llm.py | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
root@4b0009fd93b32486:~/automated-ai-research-experiments# python llm.py 
🔍 Device: CUDA
GPU: NVIDIA GeForce RTX 4090
Memory: 25.3 GB
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🧪 EXPERIMENTAL DESIGN: Routing Noise in Small-Scale MoEs
======================================================================
Conditions: 3 noise levels × 2 weight levels × 3 seeds = 18 total runs

🔬 Running experiment with seed=42, noise_std=0.0, load_balancing_weight=0.01
🌱 Set all seeds to 42
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:43<00:00,  9.14it/s, loss=5.3661, aux=0.0602, acc=0.

📊 Final Results:
   Val Loss: 4.8656
   Val Accuracy: 0.2399
   Val Perplexity: 129.75
   Best Val Perplexity: 129.75
   Expert Usage CV: 0.0661
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=43, noise_std=0.0, load_balancing_weight=0.01
🌱 Set all seeds to 43
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:41<00:00,  9.59it/s, loss=5.4697, aux=0.0600, acc=0.

📊 Final Results:
   Val Loss: 4.8794
   Val Accuracy: 0.2411
   Val Perplexity: 131.55
   Best Val Perplexity: 131.55
   Expert Usage CV: 0.0867
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=44, noise_std=0.0, load_balancing_weight=0.01
🌱 Set all seeds to 44
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:41<00:00,  9.57it/s, loss=5.3764, aux=0.0603, acc=0.

📊 Final Results:
   Val Loss: 4.8629
   Val Accuracy: 0.2403
   Val Perplexity: 129.40
   Best Val Perplexity: 129.40
   Expert Usage CV: 0.0560
✅ Experiment completed in 1.5 minutes

🔬 Running experiment with seed=42, noise_std=0.0, load_balancing_weight=0.0
🌱 Set all seeds to 42
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:42<00:00,  9.39it/s, loss=5.3738, aux=0.0000, acc=0.

📊 Final Results:
   Val Loss: 4.8751
   Val Accuracy: 0.2399
   Val Perplexity: 130.99
   Best Val Perplexity: 130.99
   Expert Usage CV: 0.2169
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=43, noise_std=0.0, load_balancing_weight=0.0
🌱 Set all seeds to 43
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:41<00:00,  9.54it/s, loss=5.4683, aux=0.0000, acc=0.

📊 Final Results:
   Val Loss: 4.8790
   Val Accuracy: 0.2419
   Val Perplexity: 131.50
   Best Val Perplexity: 131.50
   Expert Usage CV: 0.2426
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=44, noise_std=0.0, load_balancing_weight=0.0
🌱 Set all seeds to 44
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:42<00:00,  9.33it/s, loss=5.3818, aux=0.0000, acc=0.

📊 Final Results:
   Val Loss: 4.8608
   Val Accuracy: 0.2417
   Val Perplexity: 129.13
   Best Val Perplexity: 129.13
   Expert Usage CV: 0.1598
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=42, noise_std=0.1, load_balancing_weight=0.01
🌱 Set all seeds to 42
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:41<00:00,  9.55it/s, loss=5.3816, aux=0.0602, acc=0.

📊 Final Results:
   Val Loss: 4.8741
   Val Accuracy: 0.2401
   Val Perplexity: 130.86
   Best Val Perplexity: 130.86
   Expert Usage CV: 0.0620
✅ Experiment completed in 1.5 minutes

🔬 Running experiment with seed=43, noise_std=0.1, load_balancing_weight=0.01
🌱 Set all seeds to 43
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:41<00:00,  9.65it/s, loss=5.4805, aux=0.0601, acc=0.

📊 Final Results:
   Val Loss: 4.8888
   Val Accuracy: 0.2406
   Val Perplexity: 132.79
   Best Val Perplexity: 132.79
   Expert Usage CV: 0.0835
✅ Experiment completed in 1.5 minutes

🔬 Running experiment with seed=44, noise_std=0.1, load_balancing_weight=0.01
🌱 Set all seeds to 44
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:42<00:00,  9.38it/s, loss=5.3961, aux=0.0602, acc=0.

📊 Final Results:
   Val Loss: 4.8714
   Val Accuracy: 0.2405
   Val Perplexity: 130.51
   Best Val Perplexity: 130.51
   Expert Usage CV: 0.0465
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=42, noise_std=0.1, load_balancing_weight=0.0
🌱 Set all seeds to 42
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:42<00:00,  9.32it/s, loss=5.3875, aux=0.0000, acc=0.

📊 Final Results:
   Val Loss: 4.8820
   Val Accuracy: 0.2403
   Val Perplexity: 131.90
   Best Val Perplexity: 131.90
   Expert Usage CV: 0.2505
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=43, noise_std=0.1, load_balancing_weight=0.0
🌱 Set all seeds to 43
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:41<00:00,  9.55it/s, loss=5.4884, aux=0.0000, acc=0.

📊 Final Results:
   Val Loss: 4.8899
   Val Accuracy: 0.2416
   Val Perplexity: 132.94
   Best Val Perplexity: 132.94
   Expert Usage CV: 0.2667
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=44, noise_std=0.1, load_balancing_weight=0.0
🌱 Set all seeds to 44
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:42<00:00,  9.38it/s, loss=5.3912, aux=0.0000, acc=0.

📊 Final Results:
   Val Loss: 4.8698
   Val Accuracy: 0.2412
   Val Perplexity: 130.30
   Best Val Perplexity: 130.30
   Expert Usage CV: 0.1665
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=42, noise_std=0.5, load_balancing_weight=0.01
🌱 Set all seeds to 42
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:41<00:00,  9.54it/s, loss=5.5110, aux=0.0604, acc=0.

📊 Final Results:
   Val Loss: 4.9736
   Val Accuracy: 0.2346
   Val Perplexity: 144.55
   Best Val Perplexity: 144.55
   Expert Usage CV: 0.0541
✅ Experiment completed in 1.5 minutes

🔬 Running experiment with seed=43, noise_std=0.5, load_balancing_weight=0.01
🌱 Set all seeds to 43
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:41<00:00,  9.67it/s, loss=5.6111, aux=0.0605, acc=0.

📊 Final Results:
   Val Loss: 4.9835
   Val Accuracy: 0.2357
   Val Perplexity: 145.99
   Best Val Perplexity: 145.99
   Expert Usage CV: 0.0808
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=44, noise_std=0.5, load_balancing_weight=0.01
🌱 Set all seeds to 44
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:42<00:00,  9.43it/s, loss=5.5113, aux=0.0603, acc=0.

📊 Final Results:
   Val Loss: 4.9598
   Val Accuracy: 0.2371
   Val Perplexity: 142.57
   Best Val Perplexity: 142.57
   Expert Usage CV: 0.0567
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=42, noise_std=0.5, load_balancing_weight=0.0
🌱 Set all seeds to 42
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:43<00:00,  9.28it/s, loss=5.5070, aux=0.0000, acc=0.

📊 Final Results:
   Val Loss: 4.9770
   Val Accuracy: 0.2355
   Val Perplexity: 145.03
   Best Val Perplexity: 145.03
   Expert Usage CV: 0.4388
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=43, noise_std=0.5, load_balancing_weight=0.0
🌱 Set all seeds to 43
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:42<00:00,  9.49it/s, loss=5.5995, aux=0.0000, acc=0.

📊 Final Results:
   Val Loss: 4.9707
   Val Accuracy: 0.2383
   Val Perplexity: 144.13
   Best Val Perplexity: 144.13
   Expert Usage CV: 0.3138
✅ Experiment completed in 1.6 minutes

🔬 Running experiment with seed=44, noise_std=0.5, load_balancing_weight=0.0
🌱 Set all seeds to 44
📦 Loading cached data from data_cache/tokenized_data_2000_500000.pkl
✅ Loaded 2000 documents, 500,000 tokens from cache

🚀 Training MoE model with 8 experts (top-2)
  📊 Total parameters: 79,059,840
  📊 Active parameters: 22,436,736
  📊 Expert parameters: 56,623,104
  📊 Parameter efficiency: 28.4% active per forward pass
  Muon parameters: 60,180,480
  AdamW parameters: 18,879,360
Training MoE: 100%|█| 400/400 [00:42<00:00,  9.31it/s, loss=5.5089, aux=0.0000, acc=0.

📊 Final Results:
   Val Loss: 4.9679
   Val Accuracy: 0.2377
   Val Perplexity: 143.73
   Best Val Perplexity: 143.73
   Expert Usage CV: 0.2510
✅ Experiment completed in 1.6 minutes

💾 Results saved to moe_routing_noise_experiment_results.csv

📊 EXPERIMENT SUMMARY
======================================================================
Noise: 0.0, LB Weight: 0.01
  Best Val PPL: 130.23 ± 0.94
  Avg Time: 1.6 minutes
  Expert Usage CV: 0.0696

Noise: 0.0, LB Weight: 0.0
  Best Val PPL: 130.54 ± 1.02
  Avg Time: 1.6 minutes
  Expert Usage CV: 0.2064

Noise: 0.1, LB Weight: 0.01
  Best Val PPL: 131.38 ± 1.00
  Avg Time: 1.6 minutes
  Expert Usage CV: 0.0640

Noise: 0.1, LB Weight: 0.0
  Best Val PPL: 131.71 ± 1.09
  Avg Time: 1.6 minutes
  Expert Usage CV: 0.2279

Noise: 0.5, LB Weight: 0.01
  Best Val PPL: 144.37 ± 1.40
  Avg Time: 1.6 minutes
  Expert Usage CV: 0.0639

Noise: 0.5, LB Weight: 0.0
  Best Val PPL: 144.30 ± 0.54
  Avg Time: 1.6 minutes
  Expert Usage CV: 0.3345

✅ All 18 experiments completed!
root@4b0009fd93b32486:~/automated-ai-research-experiments# 