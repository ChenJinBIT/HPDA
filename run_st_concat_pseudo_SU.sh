for j in 0.0001
	do
		for i in 0.5
		do
				CUDA_VISIBLE_DEVICES=0 python train_st_concat_pseudo.py \
					--name st_concat_pseudo_vid_SU_CLASS_40_vanilla_cl_t-$i-realA_fakeB_recB_MSE_c3_d1_g6_ngf128_ncf1024_lambdaABC100_b8_e200_200_lr-$j-pseudo-50 \
					--model cycle_gan_multiD_st_concat_pseudo_causal \
					--save_epoch_freq 100 \
					--test_accuracy_freq 10 \
					--test_accuracy_freq_S 100 \
					--ngf 128 \
					--ncf 1024 \
					--c_threshold $i \
					--class_num 40 \
					--context_dim 1024 \
					--netG featurev2 \
					--netC feature_handama \
					--gan_mode vanilla \
					--init_type kaiming \
					--dataset_mode feature \
					--lambda_A 100 \
					--lambda_B 100 \
					--lambda_C 100 \
					--lr $j \
					--batch_size 8 \
					--lr_policy linear \
					--niter 200 \
					--niter_decay 200 \
					--frame_pseudo 50 \
					--dataroot /home/chenjin/HPDA/data \
					--S_list Stanford40_resnet50_list.txt \
					--T_list ucf_i3d_feature_frame_num_list.txt \
					--F_path ucf_resnet50_frame_feature \
					--result_path ./test_results_SU_st_concat_pseudo \
					--no_html \
					--no_dropout \
					>&./train_logs_SU_causal/st_concat_pseudo_vid_SU_CLASS_40_vanilla_cl-t-$i-realA_fakeB_recB_MSE_c3_d1_g6_ngf128_ncf1024_lambdaABC100_b8_e200_200_lr-$j-pseudo-50.log
		done
done