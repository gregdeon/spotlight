python spotlight/run_spotlight.py \
219 \
inference_results/fairface_val_resnet.pkl \
spotlight_results/fairface_val_resnet_0.02_spherical_1.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \

python spotlight/run_spotlight.py \
219 \
inference_results/fairface_val_resnet.pkl \
spotlight_results/fairface_val_resnet_0.02_spherical_2.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/fairface_val_resnet_0.02_spherical_1.pkl \

python spotlight/run_spotlight.py \
219 \
inference_results/fairface_val_resnet.pkl \
spotlight_results/fairface_val_resnet_0.02_spherical_3.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/fairface_val_resnet_0.02_spherical_1.pkl spotlight_results/fairface_val_resnet_0.02_spherical_2.pkl \

python spotlight/run_spotlight.py \
219 \
inference_results/fairface_val_resnet.pkl \
spotlight_results/fairface_val_resnet_0.02_spherical_4.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/fairface_val_resnet_0.02_spherical_1.pkl spotlight_results/fairface_val_resnet_0.02_spherical_2.pkl spotlight_results/fairface_val_resnet_0.02_spherical_3.pkl \

python spotlight/run_spotlight.py \
219 \
inference_results/fairface_val_resnet.pkl \
spotlight_results/fairface_val_resnet_0.02_spherical_5.pkl \
--learning_rate 1e-2 \
--lr_patience 10 \
--print_every 20 \
--device cuda:0 \
--num_steps 5000 \
--spherical \
--past_weights spotlight_results/fairface_val_resnet_0.02_spherical_1.pkl spotlight_results/fairface_val_resnet_0.02_spherical_2.pkl spotlight_results/fairface_val_resnet_0.02_spherical_3.pkl spotlight_results/fairface_val_resnet_0.02_spherical_4.pkl \
