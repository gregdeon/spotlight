python spotlight/run_spotlight.py \
539 \
inference_results/squad_val_bert.pkl \
spotlight_results/squad_val_bert_0.05_spherical_1.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--barrier_scale 10.000000 \

python spotlight/run_spotlight.py \
539 \
inference_results/squad_val_bert.pkl \
spotlight_results/squad_val_bert_0.05_spherical_2.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/squad_val_bert_0.05_spherical_1.pkl \
--barrier_scale 10.000000 \

python spotlight/run_spotlight.py \
539 \
inference_results/squad_val_bert.pkl \
spotlight_results/squad_val_bert_0.05_spherical_3.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/squad_val_bert_0.05_spherical_1.pkl spotlight_results/squad_val_bert_0.05_spherical_2.pkl \
--barrier_scale 10.000000 \

python spotlight/run_spotlight.py \
539 \
inference_results/squad_val_bert.pkl \
spotlight_results/squad_val_bert_0.05_spherical_4.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/squad_val_bert_0.05_spherical_1.pkl spotlight_results/squad_val_bert_0.05_spherical_2.pkl spotlight_results/squad_val_bert_0.05_spherical_3.pkl \
--barrier_scale 10.000000 \

python spotlight/run_spotlight.py \
539 \
inference_results/squad_val_bert.pkl \
spotlight_results/squad_val_bert_0.05_spherical_5.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/squad_val_bert_0.05_spherical_1.pkl spotlight_results/squad_val_bert_0.05_spherical_2.pkl spotlight_results/squad_val_bert_0.05_spherical_3.pkl spotlight_results/squad_val_bert_0.05_spherical_4.pkl \
--barrier_scale 10.000000 \

