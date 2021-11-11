import argparse
from typing import List
from agent import environment
import matplotlib.pyplot as plt
from utils_cam import dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def main(args):
    
#####################################
#  dataset
#####################################
    datalist = dataset(args.dir)
    TrainingSet , TestSet = train_test_split(datalist, test_size=0.1)
    # print(TrainingSet[0]['model'])
    TrainingSet = [datalist[0]]
    for i,model in enumerate(TrainingSet):
        # print(model)
        # print(model['model'])
        a = environment()
        a.set_meshes(model['model'])
        # a.create_renderer()
        # list_sample = a.get_random_sample(args.nbr_pictures)
        landmarks_dic = a.set_landmarks(model['landmarks'])
        a.generate_landmark_meshes(0.1)
        for key,value in landmarks_dic.items():
            print(key)
            print(a.get_landmarks_angle(key))
            a.plot_fig()
            images = a.get_view()
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(images.squeeze())
            plt.axis("on")
            plt.show()

        # a.generate_landmark_meshes(0.1)
        # a.set_random_rotation()
        # a.set_random_rotation()
        # a.plot_fig()
        # a.set_random_rotation()
        # a.plot_fig()
        # dic_dataset = {}
        # for i in len(list_sample):
        #     dic_dataset[list_sample[i]]=landmarks
            
        # print(landmarks)
    # train_dataloader = DataLoader(list_sample, batch_size=4, shuffle=True)
    
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.imshow(images.squeeze())
    # plt.axis("on")
    
    # a.set_random_rotation()
    # images = a.get_view()
   
    # plt.subplot(2, 2, 2)
    # plt.imshow(images.squeeze())
    # plt.axis("on")
    
    # a.set_random_rotation()
    # images = a.get_view()

    # plt.subplot(2, 2, 3)
    # plt.imshow(images.squeeze())
    # plt.axis("on")
    
    # a.set_random_rotation()
    # images = a.get_view()

    # plt.subplot(2, 2, 4)
    # plt.imshow(images.squeeze())
    # plt.axis("on")


    # plt.show()


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='separAte all the teeth from a vtk file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir', type=str, help='dataset directory', required=True)

    # parser.add_argument('--nbr_pictures',type=int,help='number of pictures per tooth', default=5)
   
    # output_params = parser.add_argument_group('Output parameters')
    # output_params.add_argument('--out', type=str, help='Output directory with all the 2D pictures')
    
    args = parser.parse_args()
    main(args)