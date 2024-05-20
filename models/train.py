import os
import argparse
import time
import numpy as np
import random
import sys

from scipy.optimize import curve_fit
from scipy import stats


sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from torchvision import transforms

import torch.backends.cudnn as cudnn

import IQADataset
import DN_PIQA




def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def performance_fit(y_label, y_output):
    y_output_logistic = fit_function(y_label, y_output)
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic-y_label) ** 2).mean())
    MAE = np.absolute((y_output_logistic-y_label)).mean()

    return PLCC, SRCC, KRCC, RMSE, MAE

def performance_fit_scene(y_label, y_output, y_scene):

    class_type = list(set(y_scene))

    srcc_scene = []
    plcc_scene = []
    krcc_scene = []
    rmse_scene = []
    mae_scene  = []

    for i_scene in class_type:
        i_scene_model_score = []
        i_scene_mos_score = []
        for i_image in range(len(y_scene)):
            if y_scene[i_image] == i_scene:
                i_scene_model_score.append(y_output[i_image])
                i_scene_mos_score.append(y_label[i_image])
        i_scene_model_score = np.array(i_scene_model_score)
        i_scene_mos_score = np.array(i_scene_mos_score)
        test_PLCC, test_SRCC, test_KRCC, test_RMSE, test_MAE = performance_fit(i_scene_mos_score, i_scene_model_score)
        srcc_scene.append(test_SRCC)
        plcc_scene.append(test_PLCC)
        krcc_scene.append(test_KRCC)
        rmse_scene.append(test_RMSE)
        mae_scene.append(test_MAE)

    SRCC = np.mean(np.array(srcc_scene))
    PLCC = np.mean(np.array(plcc_scene))
    KRCC = np.mean(np.array(krcc_scene))
    RMSE = np.mean(np.array(rmse_scene))
    MAE = np.mean(np.array(mae_scene))

    return PLCC, SRCC, KRCC, RMSE, MAE

EPS = 1e-2
esp = 1e-8


class Fidelity_Loss(torch.nn.Module):
    def __init__(self):
        super(Fidelity_Loss, self).__init__()

    def forward(self, p, g):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))



        return torch.mean(loss)



def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="In the wild Image Quality Assessment")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--num_epochs',  help='Maximum number of training epochs.', default=30, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=40, type=int)
    parser.add_argument('--resize', help='resize.', type=int)
    parser.add_argument('--crop_size', help='crop_size.',type=int)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=float, default=10)
    parser.add_argument('--snapshot', help='Path of model snapshot.', default='', type=str)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--database_dir', type=str)

    parser.add_argument('--feature_dir', type=str)
    parser.add_argument('--face_dir', type=str)

    parser.add_argument('--type', type=str)
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--pretrained_path_face', type=str)
    parser.add_argument('--model', default='ResNet', type=str)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--with_face', action='store_true')
    parser.add_argument('--print_samples', type=int, default = 50)
    parser.add_argument('--database', default='FLIVE', type=str)
    parser.add_argument('--test_method', default='five', type=str,
                        help='use the center crop or five crop to test the image')
    parser.add_argument('--loss_type', type=str, default='L2')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--n_trial', type=int, default=1)
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--num_patch', type=int, default=1)
    parser.add_argument('--gpu_ids', type=list, default=None)


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.random_seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    
    gpu = args.gpu
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    decay_interval = args.decay_interval
    decay_ratio = args.decay_ratio
    snapshot = args.snapshot
    database = args.database
    print_samples = args.print_samples
    results_path = args.results_path
    database_dir = args.database_dir
    feature_dir = args.feature_dir
    face_dir = args.face_dir
    resize = args.resize
    crop_size = args.crop_size

    n_trial = args.n_trial
    best_all = np.zeros([n_trial, 4])
    save_model_name_all = []


    if not os.path.exists(snapshot):
        os.makedirs(snapshot)
    trained_model_file = os.path.join(snapshot, 'train-ind-{}-{}.pkl'.format(database, args.model))
    
    print('The save model name is ' + trained_model_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_filename_list = './csvfiles/exp.csv'
    test_filename_list = './csvfiles/exp_test.csv'


    print(train_filename_list)
    print(test_filename_list)
    
    # load the network
    if args.model == 'DN_PIQA':
        model = DN_PIQA.PIQ_model(pretrained_path = args.pretrained_path, pretrained_path_face= args.pretrained_path_face)


    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)



    transformations_train = transforms.Compose([transforms.Resize(resize),transforms.RandomCrop(crop_size), \
        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if args.test_method == 'one':
        transformations_test = transforms.Compose([transforms.Resize(resize),transforms.CenterCrop(crop_size), \
            transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif args.test_method == 'five':
        transformations_test = transforms.Compose([transforms.Resize(resize),transforms.FiveCrop(crop_size), \
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), \
                (lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                    std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])

     
    train_dataset = IQADataset.PIQA_pair(database_dir, feature_dir, face_dir, train_filename_list, transformations_train, 'PIQ_train')
    test_dataset = IQADataset.PIQA(database_dir, feature_dir, face_dir, test_filename_list, transformations_test, 'PIQ_validation')            


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)


    criterion = Fidelity_Loss()


    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0000001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=decay_ratio)


    print("Ready to train network")

    best_test_criterion = -1  # SROCC min
    best = np.zeros(5)

    best_output = []

    n_train = len(train_dataset)
    n_test = len(test_dataset)


    for epoch in range(num_epochs):
        # train
        model.train()

        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()

        for i, (image1, image2, feature, mos, image1_second, image2_second, feature_second, mos_second) in enumerate(train_loader):
            image1 = image1.to(device)
            image2 = image2.to(device)
            feature = feature.to(device)
            mos = mos[:,np.newaxis]
            mos = mos.to(device)

            image1_second = image1_second.to(device)
            image2_second = image2_second.to(device)
            feature_second = feature_second.to(device)
            mos_second = mos_second[:,np.newaxis]
            mos_second = mos_second.to(device)
            

            # model ouput
            mos_output = model(image1, image2, feature)
            mos_output_second = model(image1_second, image2_second, feature_second)
            mos_output_diff = mos_output - mos_output_second
            constant =torch.sqrt(torch.Tensor([2])).to(device)
            p_output = 0.5 * (1 + torch.erf(mos_output_diff / constant))

            # label
            mos_diff = mos- mos_second
            p = 0.5 * (1 + torch.erf(mos_diff / constant))


            loss = criterion(p_output, p.detach())

            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())

            optimizer.zero_grad()   # clear gradients for next train
            torch.autograd.backward(loss)
            optimizer.step()

            if (i+1) % print_samples == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / print_samples
                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                    (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, \
                        avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr_current = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr_current[0]))

        # Test 
        model.eval()
        y_output = np.zeros(n_test)
        y_test = np.zeros(n_test)
        y_scene = []

        with torch.no_grad():
            for i, (image1, image2, feature, mos, scene) in enumerate(test_loader):
                if args.test_method == 'one':
                    image1 = image1.to(device)
                    image2 = image2.to(device)
                    feature = feature.to(device)
                    y_test[i] = mos.item()
                    mos = mos.to(device)

                    outputs = model(image1, image2, feature)
                    y_output[i] = outputs.item()
                    y_scene.append(scene)

        
                elif args.test_method == 'five':
                    bs, ncrops, c, h, w = image1.size()
                    feature = torch.cat([feature,feature,feature,feature,feature],dim=0)
                    y_test[i] = mos.item()
                    image1 = image1.to(device)
                    image2 = image2.to(device)
                    mos = mos.to(device)

                    outputs = model(image1.view(-1, c, h, w),image2.view(-1, c, h, w),feature)
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                    y_output[i] = outputs_avg.item()
                    y_scene.append(scene)


            test_PLCC, test_SRCC, test_KRCC, test_RMSE, test_MAE = performance_fit_scene(y_test, y_output,y_scene)
            print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}, MAE={:.4f}".format(test_SRCC, test_KRCC, test_PLCC, test_RMSE, test_MAE))

            if test_SRCC > best_test_criterion:
                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)

                save_model_name = os.path.join(args.snapshot, args.model + '_' + \
                    args.database + '_' + args.loss_type + '_NR_v' \
                        + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC))
                print("Update best model using best_val_criterion ")
                torch.save(model.module.state_dict(), save_model_name)
                old_save_name = save_model_name
                best[0:5] = [test_SRCC, test_KRCC, test_PLCC, test_RMSE, test_MAE]
                best_test_criterion = test_SRCC  # update best val SROCC
                best_output = y_output

                print("The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}, MAE={:.4f}".format(test_SRCC, test_KRCC, test_PLCC, test_RMSE, test_MAE))
    

    print("The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}, MAE={:.4f}".format(best[0], best[1], best[2], best[3], best[4]))

