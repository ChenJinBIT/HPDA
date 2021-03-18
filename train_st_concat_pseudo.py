"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import torch
import time
import os
import numpy as np
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    print("successfully loading parse")
    batch_size = opt.batch_size
    serial_batches = opt.serial_batches
    dataset,dataset_test,S_number,T_number = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    logger = SummaryWriter("./logs/"+opt.name)

    results_dir = opt.result_path
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    result_file = os.path.join(results_dir,opt.name+'.txt')
    model.set_pseudo(True)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        #print(epoch)
        if epoch==1:
            accuracy_model_T = ['real_B']
            correct_num_T = np.zeros([1])
            total_num_T = np.zeros([1])
            with open(result_file,"a") as f:
                s_context = torch.ones([opt.class_num,opt.context_dim]).to(model.device)
                t_context = torch.ones([opt.class_num,opt.context_dim]).to(model.device)
                #print(s_context.dtype)
                category_num =torch.ones([opt.class_num,1]).to(model.device)
                for k, data_test in enumerate(dataset_test):
                    model.set_input(data_test)  # unpack data from data loader
                    #print("begin testing-----------")
                    #model.test()           # run inference
                    #predicted_results = model.get_classify_results()
                    if k<T_number:
                    #if k<10:
                        #print("k=%d"%k)
                        correct_num_temp_T,total_num_temp_T,label,s_context_,t_context_=model.get_correct_num('real_B')
                        #print(s_context_.shape)
                        s_context[label]=s_context[label]+s_context_
                        t_context[label]=t_context[label]+t_context_
                        category_num[label]=category_num[label]+1
                        correct_num_T = correct_num_T + correct_num_temp_T
                        total_num_T =total_num_T + total_num_temp_T
                    else:
                        break               
                #print("total_num_T is ",total_num_T[0])
                accuracy_T = (correct_num_T/total_num_T)*100
                #print("correct_num_T",correct_num_T)
                #print("total_num_T",total_num_T)
                #print("s_context.shape:",s_context.shape)
                #print("category_num.shape",category_num.shape)
                s_context = s_context/category_num
                #print("average s_context:",s_context)
                #print("t_context:",t_context)
                t_context = t_context/category_num
                #print("average t_context:",t_context)
                for j in range(len(accuracy_model_T)):
                    print('Epoch[%s] %s:%.2f\n'%(epoch,accuracy_model_T[j],accuracy_T[j]))
                    f.writelines('Epoch[%s] %s:%.2f\n'%(epoch,accuracy_model_T[j],accuracy_T[j]))
            model.set_context(s_context,t_context)
        if epoch == opt.frame_pseudo:
            model.set_pseudo(False)
        #if epoch==1 or epoch % opt.test_accuracy_freq == 0:
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            #print(data)
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            #start_time = time.time()
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            #print("optimize_parameters:%f"%(time.time()-start_time))

            #if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            #    save_result = total_iters % opt.update_html_freq == 0
            #    model.compute_visuals()
            #    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.usetfb_current_losses(logger, total_iters, losses)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            iter_data_time = time.time()

        if epoch % opt.test_accuracy_freq_S == 0:
            accuracy_model = ['real_A','rec_A','fake_B']
            correct_num = np.zeros([len(accuracy_model)])
            total_num = np.zeros([len(accuracy_model)])
            with open(result_file,"a") as f:
                for k, data_test in enumerate(dataset_test):
                    model.set_input(data_test)  # unpack data from data loader
                    #print("begin testing-----------")
                    model.test()           # run inference
                    #predicted_results = model.get_classify_results()
                    for j in range(len(accuracy_model)):
                        correct_num_temp,total_num_temp=model.get_correct_num(accuracy_model[j])
                        correct_num[j] = correct_num[j] + correct_num_temp
                        total_num[j] =total_num[j] + total_num_temp
                accuracy = (correct_num/total_num)*100
                for j in range(len(accuracy_model)):
                    print('Epoch[%s] %s:%.2f\n'%(epoch,accuracy_model[j],accuracy[j]))
                    f.writelines('Epoch[%s] %s:%.2f\n'%(epoch,accuracy_model[j],accuracy[j])) 
        if epoch % opt.test_accuracy_freq == 0:
            accuracy_model_T = ['real_B']
            correct_num_T = np.zeros([len(accuracy_model_T)])
            total_num_T = np.zeros([len(accuracy_model_T)])
            s_context = torch.ones([opt.class_num,opt.context_dim]).to(model.device)
            t_context = torch.ones([opt.class_num,opt.context_dim]).to(model.device)
            #print(s_context.dtype)
            category_num =torch.ones([opt.class_num,1]).to(model.device)
            with open(result_file,"a") as f:
                for k, data_test in enumerate(dataset_test):
                    model.set_input(data_test)  # unpack data from data loader
                    #print("begin testing-----------")
                    model.test()           # run inference
                    #predicted_results = model.get_classify_results()
                    if k<T_number:
                    #if k<10:
                        #print("k=%d"%k)
                        for j in range(len(accuracy_model_T)):
                            if accuracy_model_T[j]=='real_B':
                                correct_num_temp_T,total_num_temp_T,label,s_context_,t_context_=model.get_correct_num(accuracy_model_T[j])
                                #print(s_context_.dtype)
                                s_context[label]=s_context[label]+s_context_
                                t_context[label]=t_context[label]+t_context_
                                category_num[label]=category_num[label]+1
                                correct_num_T[j] = correct_num_T[j] + correct_num_temp_T
                                total_num_T[j] =total_num_T[j] + total_num_temp_T
                            else:                          
                                correct_num_temp_T,total_num_temp_T=model.get_correct_num(accuracy_model_T[j])
                                correct_num_T[j] = correct_num_T[j] + correct_num_temp_T
                                total_num_T[j] =total_num_T[j] + total_num_temp_T
                    else:
                        break               
                print("total_num_T is ",total_num_T[0])
                print("correct_num_T is ",correct_num_T[0])
                s_context = s_context/category_num
                t_context = t_context/category_num
                accuracy_T = (correct_num_T/total_num_T)*100
                model.set_context(s_context,t_context)
                for j in range(len(accuracy_model_T)):
                    print('Epoch[%s] %s:%.2f\n'%(epoch,accuracy_model_T[j],accuracy_T[j]))
                    f.writelines('Epoch[%s] %s:%.2f\n'%(epoch,accuracy_model_T[j],accuracy_T[j]))
        if epoch % 50 == 0 or epoch==0:
            correct_num_T = np.zeros([1])
            total_num_T = np.zeros([1])
            with open(result_file,"a") as f:
                for k, data_test in enumerate(dataset_test):
                    model.set_input(data_test)  # unpack data from data loader
                    #print("begin testing-----------")
                    model.test()           # run inference
                    #predicted_results = model.get_classify_results()
                    if k<T_number:
                            correct_num_temp_T,total_num_temp_T=model.get_correct_num('F')
                            correct_num_T = correct_num_T + correct_num_temp_T
                            total_num_T =total_num_T + total_num_temp_T
                    else:
                        break               
                print("total_num_T is ",total_num_T[0])
                accuracy_T = (correct_num_T/total_num_T)*100
                print('Epoch[%s] F:%.2f\n'%(epoch,accuracy_T))
                f.writelines('Epoch[%s] F:%.2f\n'%(epoch,accuracy_T))
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
    logger.close()
