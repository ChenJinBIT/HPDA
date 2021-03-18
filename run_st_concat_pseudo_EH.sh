for j in 0.00005
	do
		for i in 0.5
		do
				CUDA_VISIBLE_DEVICES=1 python train_st_concat_pseudo.py \
					--name st_concat_pseudo_EH_CLASS_50_t-$i-realA_fakeB_recB_MSE_c3_d1_g4_ngf128_ncf1024_lambdaABC100_b8_e200_200_lr-$j-pseudo-200 \
					--model cycle_gan_multiD_st_concat_pseudo_causal \
					--save_epoch_freq 10 \
					--test_accuracy_freq 10 \
					--test_accuracy_freq_S 100 \
					--ngf 128 \
					--ncf 1024 \
					--c_threshold $i \
					--class_num 50 \
					--context_dim 1024 \
					--netG feature \
					--netC feature_handama \
					--gan_mode vanilla \
					--init_type kaiming \
					--dataset_mode feature \
					--lr $j \
					--lambda_A 100 \
					--lambda_B 100 \
					--lambda_C 100 \
					--batch_size 8 \
					--lr_policy linear \
					--niter 200 \
					--niter_decay 200 \
					--frame_pseudo 200 \
					--dataroot /home/chenjin/HPDA/data \
					--S_list EAD50_resnet50_feature_list.txt \
					--T_list hmdb51_i3d_feature_num_list.txt \
					--F_path hmdb51_resnet50_frame_feature \
					--result_path ./test_results_EH_st_concat_pseudo \
					--no_html \
					--no_dropout \
					>&./train_logs_EH_causal/st_concat_pseudo_EH_CLASS_50_cl_t-$i-realA_fakeB_recB_MSE_c3_d1_g4_ngf128_ncf1024_lambdaABC100_b8_e200_200_lr-$j-pseudo-200.log
		done
done