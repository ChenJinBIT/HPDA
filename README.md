# HPDA
This is the implementation of ''Spatial-temporal Causal Inference for Partial Image-to-video Adaptation (AAAI2021)'' using Pytorch. 

### Environment Python 3.6 and PyTorch 0.4.0

#### 1. Prepare data list
* source data list (put it in opt.dataroot/list/sourcelistname.txt)  
The format of sourcelistname.txt is the resnet50 feature of an image and the corresponding label    
  * For example:  
  home/chenjin/HPDA/data/Stanford_resnet50_feature/WritingOnBoard/writing_on_a_board_143.npy 11
  home/chenjin/HPDA/data/Stanford_resnet50_feature/WritingOnBoard/writing_on_a_board_170.npy 11  
  ...

* target data list (put it in opt.dataroot/list/targetlistname.txt)  
The format of targetlistname.txt is the i3d feature of a video, the corresponding label, the number of frames of this video    
  * For example:  
  home/chenjin/HPDA/data/ucf_i3d_feature/WritingOnBoard/v_WritingOnBoard_g11_c05.npy 11 74  
  home/chenjin/HPDA/data/ucf_i3d_feature/WritingOnBoard/v_WritingOnBoard_g19_c01.npy 11 74  
  ...
* Prepare video frame features (saved in dataroot/opt.F_path)  
The format of video frame features is ``opt.F_path/class_dir/video_name/framenum.npy''  

#### 2. Run
* For S-U task, bash run_st_concat_pseudo_SU.sh  
* For E-H task, bash run_st_concat_pseudo_EH.sh

### Citation
'''
@inproceedings{chen2021spatial,  
  title={Spatial-temporal Causal Inference for Partial Image-to-video Adaptation},  
  author={Chen, Jin and Wu, Xinxiao and Yao Hu and Jiebo, Luo},  
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},  
  year={2021}  
}
'''
