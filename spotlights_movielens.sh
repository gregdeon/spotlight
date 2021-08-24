python spotlight/run_spotlight.py \
1000 \
inference_results/movielens_val_deepset.pkl \
spotlight_results/movielens_val_deepset_0.05_spherical_1.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \

python spotlight/run_spotlight.py \
1000 \
inference_results/movielens_val_deepset.pkl \
spotlight_results/movielens_val_deepset_0.05_spherical_2.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/movielens_val_deepset_0.05_spherical_1.pkl \

python spotlight/run_spotlight.py \
1000 \
inference_results/movielens_val_deepset.pkl \
spotlight_results/movielens_val_deepset_0.05_spherical_3.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/movielens_val_deepset_0.05_spherical_1.pkl spotlight_results/movielens_val_deepset_0.05_spherical_2.pkl \

python spotlight/run_spotlight.py \
1000 \
inference_results/movielens_val_deepset.pkl \
spotlight_results/movielens_val_deepset_0.05_spherical_4.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/movielens_val_deepset_0.05_spherical_1.pkl spotlight_results/movielens_val_deepset_0.05_spherical_2.pkl spotlight_results/movielens_val_deepset_0.05_spherical_3.pkl \

python spotlight/run_spotlight.py \
1000 \
inference_results/movielens_val_deepset.pkl \
spotlight_results/movielens_val_deepset_0.05_spherical_5.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/movielens_val_deepset_0.05_spherical_1.pkl spotlight_results/movielens_val_deepset_0.05_spherical_2.pkl spotlight_results/movielens_val_deepset_0.05_spherical_3.pkl spotlight_results/movielens_val_deepset_0.05_spherical_4.pkl \
