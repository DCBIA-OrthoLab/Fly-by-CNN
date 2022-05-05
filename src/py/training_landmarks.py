from monai.networks.nets import UNet
import argparse
import os
import glob
import torch
from monai.config import print_config
from monai.metrics import DiceMetric
from utils_training import *
from monai.data import (DataLoader,partition_dataset)
from tqdm import tqdm


def main(args):

    model_lst = []
    landmarks_lst = []
    datalist = []

#############   Get Data   #################
    
    if args.dir:
        normpath = os.path.normpath("/".join([args.dir, '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd"]]:
                if True in [name in img_fn for name in ["_RCSeg_"]]:
                    model_lst.append(img_fn)
                else:
                    landmarks_lst.append(img_fn)
       
        # for i in model_lst:
        #     print("model_lst :",i)
        # for i in landmarks_lst:
        #     print("landmarks_lst :",i)

    if len(model_lst) != len(landmarks_lst):
        print("ERROR : Not the same number of models and landmarks file")
        return
    
    for file_id in range(0,len(model_lst)):
        data = {"model" : model_lst[file_id], "landmarks" : landmarks_lst[file_id]}
        datalist.append(data)
    
    # /print(datalist)
    # for i in datalist:
    #     print("datalist :",i)
    
    
    TrainingSet , TestSet = partition_dataset(datalist,ratios=[8,2], shuffle=True)          # define dataset

    # print("trainingset:" , TrainingSet)
    # print("testset:", TestSet)




    
    # print(train_trans[0]["model"].size())
# # #####################################
# #  Load data
# # #####################################

    train_load = Loader(TrainingSet)
    Train_loader = DataLoader(
        train_load, 
        batch_size=10, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    test_load = Loader(TestSet)
    Test_loader = DataLoader(
        test_load, 
        batch_size=10, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    # print(len(datalist))
    # print(len(Train_loader))
    # print(len(train_load))
    # print(len(Test_loader))
    # print(len(test_load))
# #####################################
#  Network, Losses, Optimizer
# #####################################

    learning_rate = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2,
        in_channels=7,
        out_channels=7,
        channels=(64, 128, 256, 512),
        strides=(2,2,2)
    ).to(device)
    loss_function = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# # # # # #############  Training ##################
    
    epoch_num = args.num_epoch
    test_interval = 10
    best_metric = 0
    best_metric_epoch = 0
    # epoch_loss_values = list()
    dice_metric = DiceMetric()
    max_training_steps = len(TrainingSet) // Train_loader.batch_size 
    global_test_steps = len(TestSet) // Test_loader.batch_size 

    for epoch in range(epoch_num):
        print("-" * 20)
        print(f"epoch {epoch + 1}/{epoch_num}")

        epoch_loss = 0
        steps = 0
        model.train()           # put the network in train mode
        epoch_iterator = tqdm(Train_loader, dynamic_ncols=True)
        
        for batch in epoch_iterator:
            if steps<max_training_steps:
                inputs, landmarks = batch["model"].to(device), batch["landmarks"].to(device)            # move the data to the GPU
                outputs = model(inputs)            # run the network forwards
                loss = loss_function(outputs, landmarks)            # run the loss function on the outputs
                optimizer.zero_grad()            # prepare the gradients for this step's back propagation  
                loss.backward()            # compute the gradients
                optimizer.step()            # tell the optimizer to update the weights according to the gradients and its internal optimisation strategy
                epoch_loss += loss.item()
                epoch_iterator.set_description(f"Training, loss:{epoch_loss}")
                steps+=1

        # epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % test_interval == 0:
            model.eval()            # switch off training features of the network for this pass
            dice_vals =list()
            epoch_iterator_val = tqdm(Test_loader, dynamic_ncols=True)
            with torch.no_grad():            # 'with torch.no_grad()' switches off gradient calculation for the scope of its context
                for steps,val_batch in enumerate(epoch_iterator_val):
                    val_images, val_landmarks = val_batch["model"].to(device), val_batch["landmarks"].to(device)
                    val_pred = model(val_images)                    # run the network
                    dice_metric(val_landmarks,val_pred)
                    dice = dice_metric.aggregate().item()
                    dice_vals.append(dice)
                    epoch_iterator_val.set_description(f"Validate ({steps}/{global_test_steps} Steps) (dice={dice})")
                    mean_dice_val = np.mean(dice_vals)
                
                if mean_dice_val > best_metric:
                    best_metric = mean_dice_val
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(args.dir, "best_metric_model.pth"))
                    print("saved new best metric network")
                    print(f"Model Was Saved ! Current Best Avg. Dice: {best_metric} at epoch: {best_metric_epoch} Current Avg. Dice: {mean_dice_val}" )
                    


    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='training a unet model to predict landmarks on model 3D of teeth', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir', type=str, help='path directory', required=True)

    input_param.add_argument('--num_epoch', type=int, help='number of epochs ', required=True)
    
    output_params = parser.add_argument_group('Output parameters')
    # output_params.add_argument('--out', type=str, help='Output directory', required=True)
   
    args = parser.parse_args()
    
    main(args)