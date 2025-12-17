
#python train_vq.py --name rvq_test3 --gpu_id 0 --dataset_name kit --batch_size 256 --num_quantizers 6  --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05
#python train_vq.py --name rvq_test2 --gpu_id 0 --dataset_name t2m --batch_size 512 --num_quantizers 6  --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05
#python train_vq.py --name rvq_kae_sad --gpu_id 0 --dataset_name kae-sad --batch_size 512 --num_quantizers 6  --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05
# python train_vq.py --name rvq_kae_sad2 --gpu_id 0 --dataset_name kae-sad --batch_size 512 --num_quantizers 6  --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05 --
# python train_vq.py --name rvq_kae_5 --gpu_id 0 --dataset_name kae --batch_size 512 --num_quantizers 6  --max_epoch 200 --quantize_dropout_prob 0.2 --gamma 0.05
# python train_vq.py --name rvq_kae2_1 --gpu_id 0 --dataset_name kae --batch_size 512 --num_quantizers 6  --max_epoch 200 --quantize_dropout_prob 0.2 --gamma 0.05

#python train_t2m_transformer.py --name mtrans_1 --gpu_id 0 --dataset_name kit --batch_size 16 --vq_name rvq_test1
#python train_t2m_transformer.py --name mtrans_2 --gpu_id 0 --dataset_name t2m --batch_size 64 --vq_name rvq_test2
#python train_t2m_transformer.py --name mtrans_kae --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_kae_3
# python train_t2m_transformer.py --name mtrans_kae_4 --gpu_id 0 --dataset_name sad --batch_size 64 --vq_name rvq_kae_4
# python train_t2m_transformer.py --name mtrans_kae_5 --gpu_id 0 --dataset_name sad --batch_size 64 --vq_name rvq_test2
# python train_t2m_transformer.py --name mtrans_kae_6 --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_test2
# python train_t2m_transformer.py --name mtrans_kae_6_2 --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_test2
# python train_t2m_transformer.py --name mtrans_kae_3 --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_kae_3
# python train_t2m_transformer.py --name mtrans_kae_8 --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_test2
# python train_t2m_transformer.py --name mtrans_kae2_1 --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_test2
# python train_t2m_transformer.py --name mtrans_kae2_2 --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_test2

#python train_res_transformer.py --name rtrans_1  --gpu_id 0 --dataset_name kit --batch_size 16 --vq_name rvq_test1 --cond_drop_prob 0.2 --share_weight
#python train_res_transformer.py --name rtrans_2  --gpu_id 0 --dataset_name t2m --batch_size 64 --vq_name rvq_test2 --cond_drop_prob 0.2 --share_weight
#python train_res_transformer.py --name rtrans_kae  --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_kae_3 --cond_drop_prob 0.2 --share_weight
#python train_res_transformer.py --name rtrans_kae_4  --gpu_id 0 --dataset_name sad --batch_size 64 --vq_name rvq_kae_4 --cond_drop_prob 0.2 --share_weight
#python train_res_transformer.py --name rtrans_kae_5  --gpu_id 0 --dataset_name sad --batch_size 64 --vq_name rvq_test2 --cond_drop_prob 0.2 --share_weight
#python train_res_transformer.py --name rtrans_kae_6_2  --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_test2 --cond_drop_prob 0.2 --share_weight
#python train_res_transformer.py --name rtrans_kae_3  --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_kae_3 --cond_drop_prob 0.2 --share_weight
# python train_res_transformer.py --name rtrans_kae_8  --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_test2 --cond_drop_prob 0.2 --share_weight
# python train_res_transformer.py --name rtrans_kae2_1  --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_test2 --cond_drop_prob 0.2 --share_weight
# python train_res_transformer.py --name rtrans_kae2_2  --gpu_id 0 --dataset_name kae --batch_size 64 --vq_name rvq_test2 --cond_drop_prob 0.2 --share_weight

#python eval_t2m_vq.py --gpu_id 0 --name rvq_test2 --dataset_name t2m --ext rvq_test2
#python eval_t2m_vq.py --gpu_id 0 --name rvq_kae_sad --dataset_name kae-sad --ext rvq_kae_sad
#python eval_t2m_trans_res.py --res_name rtrans_2 --dataset_name t2m --name mtrans_2 --gpu_id 0 --cond_scale 4 --time_steps 10 --ext eval2
#python eval_t2m_trans_res.py --res_name rtrans_2_2 --dataset_name t2m --name mtrans_2_2 --gpu_id 0 --cond_scale 4 --time_steps 10 --ext eval2_2


# python gen_t2m.py --gpu_id 0 --ext exp6 --name mtrans_kae_6_2 --dataset_name kae --res_name rtrans_kae_6 --text_prompt "A person is running on a treadmill."
# python gen_t2m.py --gpu_id 0 --ext exp6 --name mtrans_kae_3 --dataset_name kae --res_name rtrans_kae_3 --text_prompt "A person is running on a treadmill."
# python gen_t2m.py --gpu_id 0 --ext exp6 --name mtrans_kae_5 --dataset_name kae --res_name rtrans_kae_5 --text_prompt "A person is running on a treadmill."
# python gen_t2m.py --gpu_id 0 --ext exp6 --name mtrans_kae_8 --dataset_name kae --res_name rtrans_kae_8 --text_prompt "A person is running on a treadmill." --which_epoch iter_100.tar
#She walked forward slowly with her head down, her waist bent
#She walked forward sorrowfully
#She hopped forward
#she hopped-and-skipped forward