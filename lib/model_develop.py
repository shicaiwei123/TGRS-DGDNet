'''模型训练相关的函数'''

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import time
import csv
import os
from torchtoolbox.tools import mixup_criterion, mixup_data
import time

import os
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.nn.functional as F

from lib.model_develop_utils import GradualWarmupScheduler, calc_accuracy
from loss.mmd_loss import MMD_loss
from lib.processing_utils import PCA_svd


def calc_accuracy_multi(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
            batch_sample['label']
        if torch.cuda.is_available():
            img_m1 = img_m1.cuda()
            img_m2 = img_m2.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batch = model(img_m1, img_m2)
            if isinstance(outputs_batch, tuple):
                output_batch = outputs_batch[0]
                # output_batch_1 = outputs_batch[0]
                # output_batch_2 = outputs_batch[1]
                # output_batch = (output_batch_1 + output_batch_2) / 2
            else:
                output_batch = outputs_batch

            # print(output_batch.shape)
            # print(outputs_batch)
            outputs_full.append(output_batch)
            labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    predict_arr = np.array(labels_predicted.cpu())
    label_arr = np.array(labels_full.cpu())
    aa_acc = 0
    for i in range(args.class_num):
        label_position = np.where(label_arr == i, 1, 0)
        prediction_position = np.where(predict_arr == i, 1, 0)
        # print(label_position)
        # print(prediction_position)
        diff = label_position - prediction_position
        wrong_num = np.sum(diff == 1)
        all_num = np.sum(label_position == 1)
        # print(((all_num - wrong_num) / all_num))
        aa_acc = aa_acc + ((all_num - wrong_num) / all_num)
    aa_acc = aa_acc / args.class_num

    Pe = 0
    for i in range(args.class_num):
        label_position = np.where(label_arr == i, 1, 0)
        prediction_position = np.where(predict_arr == i, 1, 0)

        Pe = Pe + np.sum(label_position) * np.sum(prediction_position)

    Pe = Pe / (len(label_arr) * len(label_arr))
    ka_acc = (accuracy - Pe) / (1 - Pe)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1
        try:
            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            APCER = float("%.6f" % APCER)
            NPCER = float("%.6f" % NPCER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(e)
            return [accuracy, 1, 1, 1, 1, 1]

        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy, aa_acc, ka_acc]


def calc_accuracy_multi_tri(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_m1, img_m2, img_m3, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['m_3'], \
            batch_sample['label']
        if torch.cuda.is_available():
            img_m1 = img_m1.cuda()
            img_m2 = img_m2.cuda()
            img_m3 = img_m3.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batch = model(img_m1, img_m2, img_m3)
            # print(isinstance(outputs_batch,tuple))
            if isinstance(outputs_batch, tuple):
                output_batch = outputs_batch[0]
                # output_batch_1 = outputs_batch[0]
                # output_batch_2 = outputs_batch[1]
                # output_batch = (output_batch_1 + output_batch_2) / 2
            else:
                output_batch = outputs_batch
            # print(outputs_batch)
            outputs_full.append(output_batch)
            labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    predict_arr = np.array(labels_predicted.cpu())
    label_arr = np.array(labels_full.cpu())

    aa_acc = 0
    for i in range(args.class_num):
        label_position = np.where(label_arr == i, 1, 0)
        prediction_position = np.where(predict_arr == i, 1, 0)
        # print(label_position)
        # print(prediction_position)
        diff = label_position - prediction_position
        wrong_num = np.sum(diff == 1)
        # print(label_position)
        all_num = np.sum(label_position == 1)
        # print(all_num)
        # print(((all_num - wrong_num) / all_num))
        aa_acc = aa_acc + ((all_num - wrong_num) / all_num)
    aa_acc = aa_acc / args.class_num

    Pe = 0
    for i in range(args.class_num):
        label_position = np.where(label_arr == i, 1, 0)
        prediction_position = np.where(predict_arr == i, 1, 0)

        Pe = Pe + np.sum(label_position) * np.sum(prediction_position)

    Pe = Pe / (len(label_arr) * len(label_arr))
    ka_acc = (accuracy - Pe) / (1 - Pe)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy, aa_acc, ka_acc]


def calc_accuracy_kd(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
            batch_sample['label']

        if torch.cuda.is_available():
            img_m1 = img_m1.cuda()
            img_m2 = img_m2.cuda()
            target = target.cuda()

        if args.student_data == args.pair_modalities[0]:
            test_data = img_m1
        else:
            test_data = img_m2
        with torch.no_grad():
            outputs_batch = model(test_data)

            # 如果有多个返回值只取第一个
            if isinstance(outputs_batch, tuple):
                outputs_batch = outputs_batch[0]
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def calc_accuracy_kd_patch_feature(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
            batch_sample['label']

        if torch.cuda.is_available():
            img_m1 = img_m1.cuda()
            img_m2 = img_m2.cuda()
            target = target.cuda()

        if args.student_data == args.pair_modalities[0]:
            test_data = img_m1
        else:
            test_data = img_m2
        with torch.no_grad():
            outputs_batch = model(test_data)

            # 如果有多个返回值只取第一个
            if isinstance(outputs_batch, tuple):
                outputs_batch = outputs_batch[0]
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2
            ACER = (APCER + NPCER) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            APCER = float("%.6f" % APCER)
            NPCER = float("%.6f" % NPCER)
            ACER = float("%.6f" % ACER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER, ACER]
    else:
        return [accuracy]


def train_base_multi(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    is_rec = False

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar)
            if isinstance(outputs, tuple):
                is_rec = True
                output = outputs[0]
                x_origin = outputs[1]
                x_rec = outputs[2]
            else:
                output = outputs

            cls_loss = cost(output, target)
            loss = cls_loss
            if is_rec:
                rec_loss_1 = mse_func(x_origin, output)
                rec_loss_2 = mse_func(x_rec, output)
                rec_loss = rec_loss_1 + rec_loss_2
                loss = cls_loss + rec_loss
                rec_loss_sum += rec_loss

            train_loss += cls_loss.item()
            loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))
        train_loss = 0
        rec_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_share(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    is_rec = False

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar)
            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs

            hsi_out = outputs[-2]
            lidar_out = outputs[-1]

            cls_loss1 = cost(output, target)
            cls_loss2 = cost(hsi_out, target)
            cls_loss3 = cost(lidar_out, target)
            cls_loss = cls_loss1

            unimodal_loss = mse_func(hsi_out, lidar_out) + cls_loss2 + cls_loss3

            total_loss = cls_loss + unimodal_loss * args.labma_unimodal

            loss = total_loss

            train_loss += loss.item()
            unimodal_loss_sum += unimodal_loss.item()
            cls_sum += cls_loss.item()
            loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), cls_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_share_auxi(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    is_rec = False

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar)
            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs

            cls_loss = cost(output, target)

            hsi_out = outputs[-2]
            lidar_out = outputs[-1]

            lidar_loss = cost(lidar_out, target)
            hsi_loss = cost(hsi_out, target)

            # # print(hsi_out.shape,lidar_out.shape)
            #
            # if 1 in args.p:
            #     unimodal_loss = mse_func(hsi_out, lidar_out)
            #
            # else:
            #     unimodal_loss = torch.zeros(1)

            loss = cls_loss + (lidar_loss + hsi_loss) * 0.1

            train_loss += loss.item()
            # unimodal_loss_sum += unimodal_loss.item()
            cls_sum += cls_loss.item()
            loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), cls_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_share_tri(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    is_rec = False

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, img_dsm, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['m_3'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                img_dsm = img_dsm.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar, img_dsm)
            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs

            cls_loss = cost(output, target)

            # hsi_out = outputs[-2]
            # lidar_out = outputs[-1]
            # # print(hsi_out.shape,lidar_out.shape)
            #
            # if 1 in args.p:
            #     unimodal_loss = mse_func(hsi_out, lidar_out)
            #
            # else:
            #     unimodal_loss = torch.zeros(1)

            loss = cls_loss

            train_loss += loss.item()
            # unimodal_loss_sum += unimodal_loss.item()
            cls_sum += cls_loss.item()
            loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi_tri(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), cls_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_share_unimodal_center(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    is_rec = False
    hsi_grad_sum = 0
    lidar_grad_sum = 0
    alpha = 0.5
    beta = 0.5

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar)

            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs

            hsi_out = outputs[-2]
            lidar_out = outputs[-1]

            cls_loss1 = cost(output, target)
            cls_loss2 = cost(hsi_out, target)
            cls_loss3 = cost(lidar_out, target)
            cls_loss = cls_loss1

            unimodal_loss = mse_func(hsi_out, lidar_out) + cls_loss2 + cls_loss3

            total_loss = cls_loss + unimodal_loss * args.labma_unimodal

            train_loss += total_loss.item()
            unimodal_loss_sum += unimodal_loss.item()
            cls_sum += cls_loss1.item()
            total_loss.backward()
            optimizer.step()

            # for p in model.special_bone_hsi.parameters():
            #     hsi_grad_sum += torch.sum(torch.abs(p.grad))
            # for p in model.special_bone_lidar.parameters():
            #     lidar_grad_sum += torch.sum(torch.abs(p.grad))

        # if epoch > 0:
        #     hsi_count = 145 * 32 + 33 * 64 + 65 * 128
        #     lidar_count = 2 * 32 + 33 * 64 + 65 * 128
        #     hsi_grad_average = (hsi_grad_sum / hsi_count / len(train_loader)).cpu().detach().numpy()
        #     lidar_grad_average = (lidar_grad_sum / lidar_count / len(train_loader)).cpu().detach().numpy()
        #     print(hsi_grad_average, lidar_grad_average)
        #
        #     alpha = hsi_grad_average / (hsi_grad_average + lidar_grad_average)
        #     beta = lidar_grad_average / (hsi_grad_average + lidar_grad_average)
        #
        #     hsi_grad_sum = 0
        #     lidar_grad_sum = 0

        # testing
        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), unimodal_loss_sum / len(train_loader), cls_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        unimodal_loss_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_share_unimodal_center_no_matching(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    is_rec = False
    hsi_grad_sum = 0
    lidar_grad_sum = 0
    alpha = 0.5
    beta = 0.5

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar)

            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs

            hsi_out = outputs[-2]
            lidar_out = outputs[-1]

            cls_loss1 = cost(output, target)
            cls_loss2 = cost(hsi_out, target)
            cls_loss3 = cost(lidar_out, target)
            cls_loss = cls_loss1

            unimodal_loss = cls_loss2 + cls_loss3

            total_loss = cls_loss + unimodal_loss * 1.0

            train_loss += total_loss.item()
            unimodal_loss_sum += unimodal_loss.item()
            cls_sum += cls_loss2.item() + cls_loss3.item()
            total_loss.backward()
            optimizer.step()

            # for p in model.special_bone_hsi.parameters():
            #     hsi_grad_sum += torch.sum(torch.abs(p.grad))
            # for p in model.special_bone_lidar.parameters():
            #     lidar_grad_sum += torch.sum(torch.abs(p.grad))

        # if epoch > 0:
        #     hsi_count = 145 * 32 + 33 * 64 + 65 * 128
        #     lidar_count = 2 * 32 + 33 * 64 + 65 * 128
        #     hsi_grad_average = (hsi_grad_sum / hsi_count / len(train_loader)).cpu().detach().numpy()
        #     lidar_grad_average = (lidar_grad_sum / lidar_count / len(train_loader)).cpu().detach().numpy()
        #     print(hsi_grad_average, lidar_grad_average)
        #
        #     alpha = hsi_grad_average / (hsi_grad_average + lidar_grad_average)
        #     beta = lidar_grad_average / (hsi_grad_average + lidar_grad_average)
        #
        #     hsi_grad_sum = 0
        #     lidar_grad_sum = 0

        # testing
        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), unimodal_loss_sum / len(train_loader), cls_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        unimodal_loss_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_mmformer(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    is_rec = False

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar)

            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs

            hsi_out = outputs[-2]
            lidar_out = outputs[-1]

            cls_loss1 = cost(output, target)
            cls_loss2 = cost(hsi_out, target)
            cls_loss3 = cost(lidar_out, target)
            unimodal_loss = mse_func(hsi_out, lidar_out) + cls_loss2 + cls_loss3

            cls_loss = cls_loss1 + unimodal_loss

            total_loss = cls_loss

            train_loss += total_loss.item()
            cls_sum += cls_loss2.item() + cls_loss3.item()
            total_loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), cls_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_shaspec(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    dco_loss_sum = 0
    dao_loss_sum = 0
    is_rec = False

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            torch.autograd.set_detect_anomaly(True)

            target_predict, dco_predict, specific_feature_label, m1_feature_share_cache, m2_feature_share_cache, m1_predict, m2_predict = model(
                img_hsi, img_lidar)

            task_loss = cost(target_predict, target)
            cls_loss2 = cost(m1_predict, target)
            cls_loss3 = cost(m2_predict, target)
            unimodal_loss = mse_func(m1_predict, m2_predict) + cls_loss2 + cls_loss3

            dao_loss = mse_func(m1_feature_share_cache, m2_feature_share_cache)

            dco_loss = cost(dco_predict, specific_feature_label)

            cls_loss = task_loss + 1.0 * dao_loss + 0.02 * dco_loss + unimodal_loss * 0

            total_loss = cls_loss

            train_loss += total_loss.item()
            dao_loss_sum += dao_loss.item()
            dco_loss_sum += dco_loss.item()
            cls_sum += cls_loss2.item() + cls_loss3.item()
            total_loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), dco_loss_sum / len(train_loader), dao_loss_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        cls_sum = 0
        dao_loss_sum = 0
        dco_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_shaspec_tri(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    dco_loss_sum = 0
    dao_loss_sum = 0
    is_rec = False

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, img_dsm, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['m_3'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                img_dsm = img_dsm.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            torch.autograd.set_detect_anomaly(True)

            target_predict, dco_predict, specific_feature_label, m1_feature_share_cache, m2_feature_share_cache, m3_feature_share_cache, m1_predict, m2_predict, m3_predict = model(
                img_hsi, img_lidar, img_dsm)

            task_loss = cost(target_predict, target)
            cls_loss2 = cost(m1_predict, target)
            cls_loss3 = cost(m2_predict, target)
            cls_loss4 = cost(m3_predict, target)
            unimodal_loss = mse_func(m1_predict, m2_predict) + cls_loss2 + cls_loss3 + cls_loss4 + mse_func(m2_predict,
                                                                                                            m3_predict)

            dao_loss = mse_func(m1_feature_share_cache, m2_feature_share_cache) + mse_func(
                m2_feature_share_cache, m3_feature_share_cache)

            dco_loss = cost(dco_predict, specific_feature_label)

            cls_loss = task_loss + 1.0 * dao_loss + 0.02 * dco_loss + unimodal_loss * 0.5

            total_loss = cls_loss

            train_loss += total_loss.item()
            dao_loss_sum += dao_loss.item()
            dco_loss_sum += dco_loss.item()
            cls_sum += cls_loss2.item() + cls_loss3.item()
            total_loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi_tri(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), dco_loss_sum / len(train_loader), dao_loss_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        cls_sum = 0
        dao_loss_sum = 0
        dco_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_mmformer_tri(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    is_rec = False

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, img_dsm, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['m_3'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                img_dsm = img_dsm.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar, img_dsm)
            if isinstance(outputs, tuple):
                output = outputs[0]
                x3_out = outputs[-1]
                x2_out = outputs[-2]
                x1_out = outputs[-3]
            else:
                output = outputs

            cls_loss1 = cost(output, target)
            cls_loss2 = cost(x1_out, target)
            cls_loss3 = cost(x2_out, target)
            cls_loss4 = cost(x3_out, target)
            cls_loss = cls_loss1 + cls_loss2 + cls_loss3 + cls_loss4

            total_loss = cls_loss

            train_loss += total_loss.item()
            cls_sum += cls_loss2.item() + cls_loss3.item()
            total_loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi_tri(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), cls_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_tri(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    rec_loss_sum = 0
    cls_sum = 0
    unimodal_sum = 0
    is_rec = False

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, img_dsm, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['m_3'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                img_dsm = img_dsm.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar, img_dsm)
            if isinstance(outputs, tuple):
                output = outputs[0]
                x3_out = outputs[-1]
                x2_out = outputs[-2]
                x1_out = outputs[-3]
            else:
                output = outputs

            cls_loss1 = cost(output, target)
            cls_loss2 = cost(x1_out, target)
            cls_loss3 = cost(x2_out, target)
            cls_loss4 = cost(x3_out, target)

            cls_loss = cls_loss1 + cls_loss2 + cls_loss3 + cls_loss4

            unimodal_loss = mse_func(x1_out, x2_out) + mse_func(x2_out, x3_out)

            loss = cls_loss + unimodal_loss * args.labma_unimodal

            cls_sum += cls_loss2.item() + cls_loss3.item() + cls_loss4.item()
            unimodal_sum += unimodal_loss.item()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi_tri(model, loader=test_loader, hter=False, verbose=True, args=args)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(train_loss / len(train_loader), cls_sum / len(train_loader), unimodal_sum / len(train_loader))
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))
        train_loss = 0
        rec_loss_sum = 0
        unimodal_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_knowledge_distill(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_m1 = img_m1.cuda()
                img_m2 = img_m2.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            if '+' in args.teacher_data:  # multimodal_to_ single modality
                teacher_out = teacher_model(img_m1, img_m2)
            else:  # single to single
                if args.teacher_data == args.pair_modalities[0]:
                    teacher_out_dropout, teacher_out = teacher_model(img_m1)
                else:
                    teacher_out_dropout, teacher_out = teacher_model(img_m2)

            if args.student_data == args.pair_modalities[0]:
                student_out_dropout, student_out = student_model(img_m1)
            else:
                student_out_dropout, student_out = student_model(img_m2)

            # 蒸馏logits损失
            if args.kd_mode in ['logits', 'st']:
                kd_logits_loss = criterionKD(student_out, teacher_out.detach())
            else:
                kd_logits_loss = 0
                print("kd_Loss error")

            cls_loss = criterionCls(student_out_dropout, target)
            kd_loss = kd_logits_loss
            loss = cls_loss + kd_loss * args.lambda_kd

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_loss_sum += kd_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_knowledge_distill_fc_feature(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    kl_loss_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            student_out, x_origin, x_rec, student_feature, student_hsi, student_lidar = student_model(img_hsi,
                                                                                                      img_lidar)
            teacher_out, x2, x3, teacher_feature, teacher_hsi, teacher_lidar = teacher_model(img_hsi, img_lidar)

            cls_loss = criterionCls(student_out, target)

            rec_loss_1 = mse_loss(x_origin, student_out)
            rec_loss_2 = mse_loss(x_rec, student_out)
            rec_loss = rec_loss_1 + rec_loss_2
            cls_loss = cls_loss + rec_loss

            student_feature = student_feature.view(student_feature.shape[0], -1)
            teacher_feature = teacher_feature.view(student_feature.shape[0], -1)
            inter_loss = kd_loss_func(student_feature, teacher_feature)
            inter_loss = torch.sum(inter_loss, dim=1).mean()

            hsi_index = torch.sum(student_hsi, dim=(1, 2, 3)).bool()
            lidar_index = torch.sum(student_lidar, dim=(1, 2, 3)).bool()
            student_lidar = student_lidar[lidar_index]
            teacher_lidar = teacher_lidar[lidar_index]
            student_hsi = student_hsi[hsi_index]
            teacher_hsi = teacher_hsi[hsi_index]

            hsi_loss = kd_loss_func(student_hsi, teacher_hsi)
            hsi_loss = torch.sum(hsi_loss, dim=1).mean()

            lidar_loss = kd_loss_func(student_lidar, teacher_lidar)
            lidar_loss = torch.sum(lidar_loss, dim=1).mean()
            intra_loss = hsi_loss + lidar_loss

            # B, C = student_hsi.shape[0], student_hsi.shape[1]
            # student_hsi = student_hsi.view(B * C, -1)
            # student_lidar = student_lidar.view(B * C, -1)
            # student_diversity = torch.cosine_similarity(student_hsi, student_lidar)
            # student_diversity = student_diversity.view(B, C)
            #
            # teacher_hsi = student_hsi.view(B * C, -1)
            # teacher_lidar = student_lidar.view(B * C, -1)
            # teacher_diversity = torch.cosine_similarity(teacher_hsi, teacher_lidar)
            # teacher_diversity = teacher_diversity.view(B, C)

            # intra_loss = kd_loss_func(student_diversity, teacher_diversity)

            if args.intra:
                kd_feature_loss = intra_loss * 0.2 + inter_loss * 0.8
            else:
                kd_feature_loss = inter_loss

            if args.method == 'mse':
                kd_feature_loss = torch.mean(kd_feature_loss, dim=1)
            else:
                kd_feature_loss = kd_feature_loss
            # print(kd_feature_loss.shape)

            # print(kd_feature_loss.shape)

            if args.margin:
                teacher_whole_out_prob = F.softmax(teacher_out, dim=1)
                H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

                H_teacher_prob = H_teacher / torch.sum(H_teacher)
                kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)
            else:
                kd_feature_loss = torch.mean(kd_feature_loss)

            if epoch > 5:
                loss = cls_loss + kd_feature_loss * args.lambda_kd_feature
            else:
                loss = cls_loss

            loss = loss

            cls_loss_sum += cls_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        # testing
        student_model.p = [0, 0]
        result_test = calc_accuracy_multi(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_feature_loss_sum / (len(train_loader)),
              kl_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def get_deveristy(fm_s_input):
    D = torch.zeros(1).cuda().float()
    for i in range(fm_s_input.shape[0]):
        fm_s = fm_s_input[i, :]
        # print(fm_s.shape)
        fm_s = fm_s.view(fm_s.shape[0], -1)
        fm_s_factors = torch.sqrt(torch.sum(fm_s * fm_s, 1))
        fm_s_trans = fm_s.t()
        fm_s_trans_factors = torch.sqrt(torch.sum(fm_s_trans * fm_s_trans, 0))
        # print(fm_s.shape,fm_s_factors.shape,fm_s_trans_factors.shape)
        fm_s_normal_factors = torch.mm(fm_s_factors.unsqueeze(1), fm_s_trans_factors.unsqueeze(0))
        G_s = torch.mm(fm_s, fm_s.t())
        G_s = (G_s / fm_s_normal_factors)
        G_s = torch.mean(G_s)
        if torch.isnan(G_s):
            continue
        # print(G_s)
        D += G_s
        # print(D/)

    D = D / fm_s_input.shape[0]
    return D


def train_knowledge_distill_fc_feature_shaspec(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD, DGD, DGD_PC
    from loss.kd.at import AT
    from loss.kd.RKD import RKDLoss
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    elif args.method == 'dgd':
        kd_loss_func = DGD().cuda()
    elif args.method == 'rkd':
        kd_loss_func = RKDLoss().cuda()
    elif args.method == 'sp':
        kd_loss_func = SP(reduction='none').cuda()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    intra_loss_func = DAD().cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    kl_loss_sum = 0
    lidar_grad_sum = 0
    hsi_grad_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            s_target_predict, s_dco_predict, s_specific_feature_label, s_m1_feature_share_cache, s_fusion_feature_cache, s_m2_feature_share_cache, s_m1_predict, s_m2_predict = student_model(
                img_hsi, img_lidar)
            t_target_predict, t_dco_predict, t_specific_feature_label, t_m1_feature_share_cache, t_fusion_feature_cache, t_m2_feature_share_cache, t_m1_predict, t_m2_predict = teacher_model(
                img_hsi, img_lidar)

            task_loss = criterionCls(s_target_predict, target)
            cls_loss2 = criterionCls(s_m1_predict, target)
            cls_loss3 = criterionCls(s_m2_predict, target)
            unimodal_loss = mse_func(s_m1_predict, s_m2_predict) + cls_loss2 + cls_loss3

            dao_loss = mse_func(s_m1_feature_share_cache, s_m2_feature_share_cache)

            dco_loss = criterionCls(s_dco_predict, s_specific_feature_label)

            cls_loss = task_loss + 1.0 * dao_loss + 0.02 * dco_loss + unimodal_loss * 0.5

            cls_loss = cls_loss

            # student_feature = student_feature.view(student_feature.shape[0], -1)
            # teacher_feature = teacher_feature.view(student_feature.shape[0], -1)

            if args.method == 'mse':
                inter_loss = kd_loss_func(s_fusion_feature_cache, t_fusion_feature_cache).mean()
            elif args.method == 'rkd':
                inter_loss = kd_loss_func(s_fusion_feature_cache, t_fusion_feature_cache)
            elif args.method == 'dgd':
                inter_loss = kd_loss_func(s_fusion_feature_cache, t_fusion_feature_cache)
            else:
                inter_loss = kd_loss_func(s_fusion_feature_cache, t_fusion_feature_cache)
                inter_loss = torch.sum(inter_loss, dim=1).mean()

            intra_loss = torch.zeros(1).float().cuda()

            if args.intra:
                kd_feature_loss = intra_loss * 0.5 + inter_loss * 0.5
            else:
                kd_feature_loss = inter_loss

            kl_loss_num = kl_loss(s_fusion_feature_cache, t_fusion_feature_cache)

            if args.margin:
                teacher_whole_out_prob = F.softmax(t_target_predict, dim=1)
                H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

                H_teacher_prob = H_teacher / torch.sum(H_teacher)
                kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)
            else:
                kd_feature_loss = torch.mean(kd_feature_loss)

            # if epoch > 20:
            #     loss = cls_loss + kd_feature_loss * args.lambda_kd_feature
            # else:
            #     loss = cls_loss

            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature

            # loss = loss + kl_loss_num * args.lambda_center_loss

            cls_loss_sum += cls_loss.item()
            kl_loss_sum += kl_loss_num.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()

            # hsi_count = 32 * (144 + 32 + 64) * 9
            # for p in student_model.special_bone_hsi.parameters():
            #     grad = p.grad
            #     grad = torch.sum(grad)
            #     hsi_grad_sum += grad
            #
            # lidar_count = 32 * (1 + 32 + 64) * 9
            # for p in student_model.special_bone_lidar.parameters():
            #     grad = p.grad
            #     grad = torch.sum(grad)
            #     lidar_grad_sum += grad

            optimizer.step()
        # if epoch > 0:
        #     print((hsi_grad_sum / hsi_count / len(train_loader)).cpu().detach().numpy(),
        #           (lidar_grad_sum / lidar_count / len(train_loader)).cpu().detach().numpy())
        #     hsi_grad_sum = 0
        #     lidar_grad_sum = 0
        # testing
        student_model.p = [0, 0]
        result_test = calc_accuracy_multi(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print("cls", cls_loss_sum / len(train_loader), "fkd", kd_feature_loss_sum / (len(train_loader)), 'lkd',
              kl_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_knowledge_distill_fc_feature_shaspec_center(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD, DGD, DGD_PC
    from loss.kd.at import AT
    from loss.kd.RKD import RKDLoss
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    elif args.method == 'dgd':
        kd_loss_func = DGD().cuda()
    elif args.method == 'rkd':
        kd_loss_func = RKDLoss().cuda()
    elif args.method == 'sp':
        kd_loss_func = SP(reduction='none').cuda()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    intra_loss_func = DAD().cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    kl_loss_sum = 0
    lidar_grad_sum = 0
    hsi_grad_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            s_target_predict, s_dco_predict, s_specific_feature_label, s_m1_feature_share_cache, s_fusion_feature_cache, s_m2_feature_share_cache, s_m1_predict, s_m2_predict = student_model(
                img_hsi, img_lidar)
            t_target_predict, t_dco_predict, t_specific_feature_label, t_m1_feature_share_cache, t_fusion_feature_cache, t_m2_feature_share_cache, t_m1_predict, t_m2_predict = teacher_model(
                img_hsi, img_lidar)

            task_loss = criterionCls(s_target_predict, target)
            cls_loss2 = criterionCls(s_m1_predict, target)
            cls_loss3 = criterionCls(s_m2_predict, target)
            unimodal_loss = mse_func(s_m1_predict, s_m2_predict) + cls_loss2 + cls_loss3

            dao_loss = mse_func(s_m1_feature_share_cache, s_m2_feature_share_cache)

            dco_loss = criterionCls(s_dco_predict, s_specific_feature_label)

            cls_loss = task_loss + 1.0 * dao_loss + 0.02 * dco_loss + unimodal_loss * 0.5

            cls_loss = cls_loss

            # student_feature = student_feature.view(student_feature.shape[0], -1)
            # teacher_feature = teacher_feature.view(student_feature.shape[0], -1)

            if args.method == 'mse':
                inter_loss = kd_loss_func(s_fusion_feature_cache, t_fusion_feature_cache).mean()
            elif args.method == 'rkd':
                inter_loss = kd_loss_func(s_fusion_feature_cache, t_fusion_feature_cache)
            elif args.method == 'dgd':
                inter_loss = kd_loss_func(s_fusion_feature_cache, t_fusion_feature_cache)
            else:
                inter_loss = kd_loss_func(s_fusion_feature_cache, t_fusion_feature_cache)
                inter_loss = torch.sum(inter_loss, dim=1).mean()

            intra_loss = torch.zeros(1).float().cuda()

            if args.intra:
                kd_feature_loss = intra_loss * 0.5 + inter_loss * 0.5
            else:
                kd_feature_loss = inter_loss

            kl_loss_num = kl_loss(s_fusion_feature_cache, t_fusion_feature_cache)

            if args.margin:
                teacher_whole_out_prob = F.softmax(t_target_predict, dim=1)
                H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

                H_teacher_prob = H_teacher / torch.sum(H_teacher)
                kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)
            else:
                kd_feature_loss = torch.mean(kd_feature_loss)

            # if epoch > 20:
            #     loss = cls_loss + kd_feature_loss * args.lambda_kd_feature
            # else:
            #     loss = cls_loss

            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature

            # loss = loss + kl_loss_num * args.lambda_center_loss

            cls_loss_sum += cls_loss.item()
            kl_loss_sum += kl_loss_num.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()

            # hsi_count = 32 * (144 + 32 + 64) * 9
            # for p in student_model.special_bone_hsi.parameters():
            #     grad = p.grad
            #     grad = torch.sum(grad)
            #     hsi_grad_sum += grad
            #
            # lidar_count = 32 * (1 + 32 + 64) * 9
            # for p in student_model.special_bone_lidar.parameters():
            #     grad = p.grad
            #     grad = torch.sum(grad)
            #     lidar_grad_sum += grad

            optimizer.step()
        # if epoch > 0:
        #     print((hsi_grad_sum / hsi_count / len(train_loader)).cpu().detach().numpy(),
        #           (lidar_grad_sum / lidar_count / len(train_loader)).cpu().detach().numpy())
        #     hsi_grad_sum = 0
        #     lidar_grad_sum = 0
        # testing
        student_model.p = [0, 0]
        result_test = calc_accuracy_multi(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print("cls", cls_loss_sum / len(train_loader), "fkd", kd_feature_loss_sum / (len(train_loader)), 'lkd',
              kl_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_knowledge_distill_fc_feature_share(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD, DGD, DGD_PC
    from loss.kd.at import AT
    from loss.kd.RKD import RKDLoss
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    elif args.method == 'dgd':
        kd_loss_func = DGD().cuda()
    elif args.method == 'rkd':
        kd_loss_func = RKDLoss().cuda()
    elif args.method == 'sp':
        kd_loss_func = SP(reduction='none').cuda()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    intra_loss_func = DAD().cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    kl_loss_sum = 0
    lidar_grad_sum = 0
    hsi_grad_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            student_out, student_feature, p, student_hsi, student_lidar = student_model(img_hsi, img_lidar)
            teacher_out, teacher_feature, p, teacher_hsi, teacher_lidar = teacher_model(img_hsi, img_lidar)

            cls_loss = criterionCls(student_out, target)

            cls_loss = cls_loss

            # student_feature = student_feature.view(student_feature.shape[0], -1)
            # teacher_feature = teacher_feature.view(student_feature.shape[0], -1)

            if args.method == 'mse':
                inter_loss = kd_loss_func(student_feature, teacher_feature).mean()
            elif args.method == 'rkd':
                inter_loss = kd_loss_func(student_feature, teacher_feature)
            elif args.method == 'dgd':
                inter_loss = kd_loss_func(student_feature, teacher_feature)
            else:
                inter_loss = kd_loss_func(student_feature, teacher_feature)
                inter_loss = torch.sum(inter_loss, dim=1).mean()

            # hsi_index = torch.sum(student_hsi, dim=(1, 2, 3)).bool()
            # lidar_index = torch.sum(student_lidar, dim=(1, 2, 3)).bool()
            # student_lidar = student_lidar[lidar_index]
            # teacher_lidar = teacher_lidar[lidar_index]
            # student_hsi = student_hsi[hsi_index]
            # teacher_hsi = teacher_hsi[hsi_index]
            #
            # # print(11111111111111111)
            #
            # hsi_loss = get_deveristy(student_hsi)
            # # print(22222222222222222222)/////
            # # hsi_loss = torch.sum(hsi_loss, dim=1).mean()
            #
            # # lidar_loss = get_deveristy(student_lidar)
            # # lidar_loss = torch.sum(lidar_loss, dim=1).mean()
            # intra_loss = hsi_loss
            # # print(intra_loss)

            intra_loss = torch.zeros(1).float().cuda()

            if args.intra:
                kd_feature_loss = intra_loss * 0.5 + inter_loss * 0.5
            else:
                kd_feature_loss = inter_loss

            kl_loss_num = kl_loss(student_feature, teacher_feature)

            if args.margin:
                teacher_whole_out_prob = F.softmax(teacher_out, dim=1)
                H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

                H_teacher_prob = H_teacher / torch.sum(H_teacher)
                kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)
            else:
                kd_feature_loss = torch.mean(kd_feature_loss)

            # if epoch > 20:
            #     loss = cls_loss + kd_feature_loss * args.lambda_kd_feature
            # else:
            #     loss = cls_loss

            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature

            # loss = loss + kl_loss_num * args.lambda_center_loss

            cls_loss_sum += cls_loss.item()
            kl_loss_sum += kl_loss_num.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()

            # hsi_count = 32 * (144 + 32 + 64) * 9
            # for p in student_model.special_bone_hsi.parameters():
            #     grad = p.grad
            #     grad = torch.sum(grad)
            #     hsi_grad_sum += grad
            #
            # lidar_count = 32 * (1 + 32 + 64) * 9
            # for p in student_model.special_bone_lidar.parameters():
            #     grad = p.grad
            #     grad = torch.sum(grad)
            #     lidar_grad_sum += grad

            optimizer.step()
        # if epoch > 0:
        #     print((hsi_grad_sum / hsi_count / len(train_loader)).cpu().detach().numpy(),
        #           (lidar_grad_sum / lidar_count / len(train_loader)).cpu().detach().numpy())
        #     hsi_grad_sum = 0
        #     lidar_grad_sum = 0
        # testing
        student_model.p = [0, 0]
        result_test = calc_accuracy_multi(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print("cls", cls_loss_sum / len(train_loader), "fkd", kd_feature_loss_sum / (len(train_loader)), 'lkd',
              kl_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_knowledge_distill_fc_feature_share_unimodal_center(net_dict, cost_dict, optimizer, train_loader, test_loader,
                                                             args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.sp import SP, DAD
    from loss.kd.RKD import RKDLoss
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']



    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    elif args.method == 'sp':
        kd_loss_func = SP(reduction='none').cuda()
    elif args.method == 'rkd':
        kd_loss_func = RKDLoss()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    intra_loss_func = DAD().cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    unimodal_loss_sum = 0
    lidar_grad_sum = 0
    hsi_grad_sum = 0
    alpha = 0.5
    beta = 0.5

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            student_out, student_feature, p, student_hsi, student_lidar = student_model(img_hsi, img_lidar)
            teacher_out, teacher_feature, p, teacher_hsi, teacher_lidar = teacher_model(img_hsi, img_lidar)

            cls_loss1 = criterionCls(student_out, target)
            cls_loss2 = criterionCls(student_hsi, target)
            cls_loss3 = criterionCls(student_lidar, target)

            cls_loss = cls_loss1

            student_feature = student_feature.view(student_feature.shape[0], -1)
            teacher_feature = teacher_feature.view(student_feature.shape[0], -1)

            if args.method == 'mse':
                inter_loss = kd_loss_func(student_feature, teacher_feature).mean()
            elif args.method == 'rkd':
                inter_loss = kd_loss_func(student_feature, teacher_feature)
            else:
                inter_loss = kd_loss_func(student_feature, teacher_feature)
                inter_loss = torch.sum(inter_loss, dim=1).mean()

            unimodal_loss = cls_loss2 + cls_loss3

            kd_feature_loss = inter_loss


            kd_feature_loss = torch.mean(kd_feature_loss)

            # print(cls_loss, kd_feature_loss, unimodal_loss)
            if torch.isnan(kd_feature_loss):
                kd_feature_loss = torch.zeros(1).cuda().float()
            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature + unimodal_loss * args.lambda_center_loss

            cls_loss_sum += cls_loss1.item()
            unimodal_loss_sum += unimodal_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()

            optimizer.step()

        #     for p in student_model.special_bone_hsi.parameters():
        #         hsi_grad_sum += torch.sum(torch.abs(p.grad))
        #     for p in student_model.special_bone_lidar.parameters():
        #         lidar_grad_sum += torch.sum(torch.abs(p.grad))
        #
        # if epoch > 0:
        #     hsi_count = 145 * 32 + 33 * 64 + 65 * 128
        #     lidar_count = 2 * 32 + 33 * 64 + 65 * 128
        #     hsi_grad_average = (hsi_grad_sum / hsi_count / len(train_loader)).cpu().detach().numpy()
        #     lidar_grad_average = (lidar_grad_sum / lidar_count / len(train_loader)).cpu().detach().numpy()
        #     print(hsi_grad_average, lidar_grad_average)
        #
        #     alpha = hsi_grad_average / (hsi_grad_average + lidar_grad_average)
        #     beta = lidar_grad_average / (hsi_grad_average + lidar_grad_average)
        #
        #     hsi_grad_sum = 0
        #     lidar_grad_sum = 0
        # testing
        student_model.p = [0, 0]
        result_test = calc_accuracy_multi(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print("cls", cls_loss_sum / len(train_loader), "lkd", kd_feature_loss_sum / (len(train_loader)), 'unimodal',
              unimodal_loss_sum / (len(train_loader)))

        with open("cls_loss_sum.csv", 'a+') as f:
            w = csv.writer(f)
            w.writerow([cls_loss_sum / len(train_loader)])
        with open("lkd_loss_sum.csv", 'a+') as f:
            w = csv.writer(f)
            w.writerow([kd_feature_loss_sum / len(train_loader)])

        with open("unimodal_loss_sum.csv", 'a+') as f:
            w = csv.writer(f)
            w.writerow([unimodal_loss_sum / len(train_loader)])

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0
        unimodal_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_knowledge_distill_fc_feature_share_tri(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    from loss.kd.RKD import RKDLoss
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    elif args.method == 'rkd':
        kd_loss_func = RKDLoss()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    intra_loss_func = DAD().cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    kl_loss_sum = 0
    lidar_grad_sum = 0
    hsi_grad_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, img_dsm, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['m_3'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                img_dsm = img_dsm.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            student_out, student_feature, p, student_hsi, student_lidar, student_dsm = student_model(img_hsi, img_lidar,
                                                                                                     img_dsm)
            teacher_out, teacher_feature, p, teacher_hsi, teacher_lidar, teacher_dsm = teacher_model(img_hsi, img_lidar,
                                                                                                     img_dsm)

            cls_loss = criterionCls(student_out, target)

            cls_loss = cls_loss

            student_feature = student_feature.view(student_feature.shape[0], -1)
            teacher_feature = teacher_feature.view(student_feature.shape[0], -1)

            if args.method == 'mse':
                inter_loss = kd_loss_func(student_feature, teacher_feature).mean()
            elif args.method == 'rkd':
                inter_loss = kd_loss_func(student_feature, teacher_feature)
            else:
                inter_loss = kd_loss_func(student_feature, teacher_feature)
                inter_loss = torch.sum(inter_loss, dim=1).mean()

            kd_feature_loss = inter_loss

            kl_loss_num = kl_loss(student_feature, teacher_feature)

            if args.margin:
                teacher_whole_out_prob = F.softmax(teacher_out, dim=1)
                H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

                H_teacher_prob = H_teacher / torch.sum(H_teacher)
                kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)
            else:
                kd_feature_loss = torch.mean(kd_feature_loss)

            if epoch > 20:
                loss = cls_loss + kd_feature_loss * args.lambda_kd_feature
            else:
                loss = cls_loss

            loss = loss + kl_loss_num * args.lambda_center_loss

            cls_loss_sum += cls_loss.item()
            kl_loss_sum += kl_loss_num.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()

            # hsi_count = 32 * (144 + 32 + 64) * 9
            # for p in student_model.special_bone_hsi.parameters():
            #     grad = p.grad
            #     grad = torch.sum(grad)
            #     hsi_grad_sum += grad
            #
            # lidar_count = 32 * (1 + 32 + 64) * 9
            # for p in student_model.special_bone_lidar.parameters():
            #     grad = p.grad
            #     grad = torch.sum(grad)
            #     lidar_grad_sum += grad

            optimizer.step()
        # if epoch > 0:
        #     print((hsi_grad_sum / hsi_count / len(train_loader)).cpu().detach().numpy(),
        #           (lidar_grad_sum / lidar_count / len(train_loader)).cpu().detach().numpy())
        #     hsi_grad_sum = 0
        #     lidar_grad_sum = 0
        # testing
        student_model.p = [0, 0, 0]
        result_test = calc_accuracy_multi_tri(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print("cls", cls_loss_sum / len(train_loader), "fkd", kd_feature_loss_sum / (len(train_loader)), 'lkd',
              kl_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_knowledge_distill_fc_feature_shaspec_tri(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    from loss.kd.RKD import RKDLoss
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    elif args.method == 'rkd':
        kd_loss_func = RKDLoss()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    intra_loss_func = DAD().cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    kl_loss_sum = 0
    lidar_grad_sum = 0
    hsi_grad_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, img_dsm, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['m_3'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                img_dsm = img_dsm.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            s_target_predict, s_dco_predict, s_specific_feature_label, s_m1_feature_share_cache, s_m2_feature_share_cache, s_m3_feature_share_cache, s_fusion_feature, s_m1_predict, s_m2_predict, s_m3_predict = student_model(
                img_hsi, img_lidar,
                img_dsm)
            t_target_predict, t_dco_predict, t_specific_feature_label, t_m1_feature_share_cache, t_m2_feature_share_cache, t_m3_feature_share_cache, t_fusion_feature, t_m1_predict, t_m2_predict, t_m3_predict = teacher_model(
                img_hsi, img_lidar,
                img_dsm)

            task_loss = criterionCls(s_target_predict, target)
            cls_loss2 = criterionCls(s_m1_predict, target)
            cls_loss3 = criterionCls(s_m2_predict, target)
            cls_loss4 = criterionCls(s_m3_predict, target)
            unimodal_loss = F.mse_loss(s_m1_predict, s_m2_predict) + cls_loss2 + cls_loss3 + cls_loss4 + F.mse_loss(
                s_m1_predict, s_m3_predict)

            dao_loss = F.mse_loss(s_m1_feature_share_cache, s_m2_feature_share_cache) + F.mse_loss(
                s_m2_feature_share_cache, s_m3_feature_share_cache)

            dco_loss = criterionCls(s_dco_predict, s_specific_feature_label)

            cls_loss = task_loss + 1.0 * dao_loss + 0.04 * dco_loss + unimodal_loss * 0.5

            cls_loss = cls_loss

            student_feature = s_fusion_feature.view(s_fusion_feature.shape[0], -1)
            teacher_feature = t_fusion_feature.view(t_fusion_feature.shape[0], -1)

            if args.method == 'mse':
                inter_loss = kd_loss_func(student_feature, teacher_feature).mean()
            elif args.method == 'rkd':
                inter_loss = kd_loss_func(student_feature, teacher_feature)
            else:
                inter_loss = kd_loss_func(student_feature, teacher_feature)
                inter_loss = torch.sum(inter_loss, dim=1).mean()

            kd_feature_loss = inter_loss

            kl_loss_num = kl_loss(student_feature, teacher_feature)

            if args.margin:
                teacher_whole_out_prob = F.softmax(t_target_predict, dim=1)
                H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

                H_teacher_prob = H_teacher / torch.sum(H_teacher)
                kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)
            else:
                kd_feature_loss = torch.mean(kd_feature_loss)

            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature

            cls_loss_sum += cls_loss.item()
            kl_loss_sum += kl_loss_num.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()

            # hsi_count = 32 * (144 + 32 + 64) * 9
            # for p in student_model.special_bone_hsi.parameters():
            #     grad = p.grad
            #     grad = torch.sum(grad)
            #     hsi_grad_sum += grad
            #
            # lidar_count = 32 * (1 + 32 + 64) * 9
            # for p in student_model.special_bone_lidar.parameters():
            #     grad = p.grad
            #     grad = torch.sum(grad)
            #     lidar_grad_sum += grad

            optimizer.step()
        # if epoch > 0:
        #     print((hsi_grad_sum / hsi_count / len(train_loader)).cpu().detach().numpy(),
        #           (lidar_grad_sum / lidar_count / len(train_loader)).cpu().detach().numpy())
        #     hsi_grad_sum = 0
        #     lidar_grad_sum = 0
        # testing
        student_model.p = [0, 0, 0]
        result_test = calc_accuracy_multi_tri(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print("cls", cls_loss_sum / len(train_loader), "fkd", kd_feature_loss_sum / (len(train_loader)), 'lkd',
              kl_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_knowledge_distill_fc_feature_share_tri_unimodal_center(net_dict, cost_dict, optimizer, train_loader,
                                                                 test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    from loss.kd.RKD import RKDLoss
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    elif args.method == 'rkd':
        kd_loss_func = RKDLoss()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    intra_loss_func = DAD().cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    kl_loss_sum = 0
    lidar_grad_sum = 0
    hsi_grad_sum = 0

    aa = 1 / 3
    bb = 1 / 3
    cc = 1 / 3

    m1_grad_sum = 0
    m2_grad_sum = 0
    m3_grad_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, img_dsm, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['m_3'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                img_dsm = img_dsm.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            student_out, student_feature, p, student_hsi, student_lidar, student_dsm = student_model(img_hsi, img_lidar,
                                                                                                     img_dsm)
            teacher_out, teacher_feature, p, teacher_hsi, teacher_lidar, teacher_dsm = teacher_model(img_hsi, img_lidar,
                                                                                                     img_dsm)

            cls_loss1 = criterionCls(student_out, target)
            cls_loss2 = criterionCls(student_hsi, target)
            cls_loss3 = criterionCls(student_lidar, target)
            cls_loss4 = criterionCls(student_dsm, target)

            cls_loss = cls_loss1

            student_feature = student_feature.view(student_feature.shape[0], -1)
            teacher_feature = teacher_feature.view(student_feature.shape[0], -1)

            if args.method == 'mse':
                inter_loss = kd_loss_func(student_feature, teacher_feature).mean()
            elif args.method == 'rkd':
                inter_loss = kd_loss_func(student_feature, teacher_feature)
            else:
                inter_loss = kd_loss_func(student_feature, teacher_feature)
                inter_loss = torch.sum(inter_loss, dim=1).mean()

            kd_feature_loss = inter_loss

            unimodal_loss = mse_loss(student_hsi, student_lidar) + mse_loss(student_lidar,
                                                                            student_dsm) + cls_loss2 + cls_loss3 + cls_loss4

            if args.margin:
                teacher_whole_out_prob = F.softmax(teacher_out, dim=1)
                H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

                H_teacher_prob = H_teacher / torch.sum(H_teacher)
                kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)
            else:
                kd_feature_loss = torch.mean(kd_feature_loss)

            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature + unimodal_loss * args.lambda_center_loss

            cls_loss_sum += cls_loss2.item() + cls_loss3.item() + cls_loss4.item()
            kl_loss_sum += unimodal_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()

            for p in student_model.modality_1.parameters():
                m1_grad_sum += torch.sum(torch.abs(p.grad))
            for p in student_model.modality_2.parameters():
                m2_grad_sum += torch.sum(torch.abs(p.grad))
            for p in student_model.modality_3.parameters():
                m3_grad_sum += torch.sum(torch.abs(p.grad))

            optimizer.step()

        if epoch > 0:
            m1_count = 181 * 32 + 33 * 64 + 65 * 128
            m2_count = 5 * 32 + 33 * 64 + 65 * 128
            m3_count = 2 * 32 + 33 * 64 + 65 * 128
            m1_grad_average = (m1_grad_sum / m1_count / len(train_loader)).cpu().detach().numpy()
            m2_grad_average = (m2_grad_sum / m2_count / len(train_loader)).cpu().detach().numpy()
            m3_grad_average = (m3_grad_sum / m3_count / len(train_loader)).cpu().detach().numpy()
            print(m1_grad_average, m2_grad_average, m3_grad_average)

            aa = m3_grad_average / (m1_grad_average + m2_grad_average + m3_grad_average)
            bb = m2_grad_average / (m1_grad_average + m2_grad_average + m3_grad_average)
            cc = m1_grad_average / (m1_grad_average + m2_grad_average + m3_grad_average)

            m1_grad_sum = 0
            m2_grad_sum = 0
            m3_grad_sum = 0
        # testing
        student_model.p = [0, 0, 0]
        result_test = calc_accuracy_multi_tri(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print("cls", cls_loss_sum / len(train_loader), "fkd", kd_feature_loss_sum / (len(train_loader)), 'ucl',
              kl_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_knowledge_distill_fc_feature_auxi(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    x_lidar_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    kl_loss_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            student_out, x_origin, x_rec, student_feature, lidar_out, p = student_model(img_hsi, img_lidar)
            teacher_out, x2, x3, teacher_feature = teacher_model(img_hsi, img_lidar)

            cls_loss = criterionCls(student_out, target)

            rec_loss_1 = mse_loss(x_origin, student_out)
            rec_loss_2 = mse_loss(x_rec, student_out)
            rec_loss = rec_loss_1 + rec_loss_2
            cls_loss = cls_loss + rec_loss

            student_feature = student_feature.view(student_feature.shape[0], -1)
            teacher_feature = teacher_feature.view(student_feature.shape[0], -1)
            kd_feature_loss = kd_loss_func(student_feature, teacher_feature)

            kl_loss_num = kl_loss(student_feature, teacher_feature)

            lidar_loss_batch = auxi_cross_entropy(lidar_out, target)
            x_lidar_loss = torch.sum(lidar_loss_batch * ((1 - p[:, 0]) * (p[:, 1]))) / p.shape[0]

            if args.method == 'mse':
                kd_feature_loss = torch.mean(kd_feature_loss, dim=1)
            else:
                kd_feature_loss = torch.sum(kd_feature_loss, dim=1)
            # print(kd_feature_loss.shape)

            # print(kd_feature_loss.shape)

            teacher_whole_out_prob = F.softmax(teacher_out, dim=1)
            H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

            H_teacher_prob = H_teacher / torch.sum(H_teacher)
            kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)

            if epoch > 5:
                loss = cls_loss + kd_feature_loss * args.lambda_kd_feature + x_lidar_loss * 0.5
            else:
                loss = cls_loss

            cls_loss_sum += cls_loss.item()
            kl_loss_sum += kl_loss_num.item()
            x_lidar_loss_sum += x_lidar_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        # testing
        student_model.p = [0, 0]
        result_test = calc_accuracy_multi(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_feature_loss_sum / (len(train_loader)),
              kl_loss_sum / (len(train_loader)), x_lidar_loss_sum / (len(train_loader)), )

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0
        x_lidar_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_share_mmanet(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    x_lidar_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    kl_loss_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            student_out, student_feature, p, hsi_out, lidar_out = student_model(img_hsi, img_lidar)
            teacher_out, teacher_feature, _, _, _ = teacher_model(img_hsi, img_lidar)

            cls_loss = criterionCls(student_out, target)

            student_feature = student_feature.view(student_feature.shape[0], -1)
            teacher_feature = teacher_feature.view(student_feature.shape[0], -1)
            kd_feature_loss = kd_loss_func(student_feature, teacher_feature)

            kl_loss_num = kl_loss(student_feature, teacher_feature)

            lidar_loss_batch = auxi_cross_entropy(lidar_out, target)
            x_lidar_loss = torch.sum(lidar_loss_batch * ((1 - p[:, 0]) * (p[:, 1]))) / p.shape[0]

            if args.method == 'mse':
                kd_feature_loss = torch.mean(kd_feature_loss, dim=1)
            else:
                kd_feature_loss = torch.sum(kd_feature_loss, dim=1)
            # print(kd_feature_loss.shape)

            # print(kd_feature_loss.shape)

            teacher_whole_out_prob = F.softmax(teacher_out, dim=1)
            H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

            H_teacher_prob = H_teacher / torch.sum(H_teacher)
            kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)

            if epoch > 5:
                loss = cls_loss + kd_feature_loss * args.lambda_kd_feature + x_lidar_loss * 0.5
            else:
                loss = cls_loss

            cls_loss_sum += cls_loss.item()
            kl_loss_sum += kl_loss_num.item()
            x_lidar_loss_sum += x_lidar_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        # testing
        student_model.p = [0, 0]
        result_test = calc_accuracy_multi(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_feature_loss_sum / (len(train_loader)),
              kl_loss_sum / (len(train_loader)), x_lidar_loss_sum / (len(train_loader)), )

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0
        x_lidar_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_share_mmanet_tri(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    from loss.kd.st import SoftTarget
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    mse_loss = nn.MSELoss().cuda()

    kl_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    x_lidar_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    kl_loss_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, img_dsm, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['m_3'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                img_dsm = img_dsm.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            student_out, student_feature, p, student_hsi, student_lidar, student_dsm = student_model(img_hsi, img_lidar,
                                                                                                     img_dsm)
            teacher_out, teacher_feature, p, teacher_hsi, teacher_lidar, teacher_dsm = teacher_model(img_hsi, img_lidar,
                                                                                                     img_dsm)

            cls_loss = criterionCls(student_out, target)

            student_feature = student_feature.view(student_feature.shape[0], -1)
            teacher_feature = teacher_feature.view(student_feature.shape[0], -1)
            kd_feature_loss = kd_loss_func(student_feature, teacher_feature)

            kl_loss_num = kl_loss(student_feature, teacher_feature)

            lidar_loss_batch = auxi_cross_entropy(student_lidar, target)
            dsm_loss_btach = auxi_cross_entropy(student_dsm, target)
            hsi_loss_batch = auxi_cross_entropy(student_hsi, target)

            # p = p.view(p.shape[0], -1)
            x_lidar_loss = torch.sum(lidar_loss_batch * ((1 - p[:, 0]) * (p[:, 1]) * ((1 - p[:, 2])))) / p.shape[0]
            x_dsm_loss = torch.sum(dsm_loss_btach * ((1 - p[:, 0]) * (1 - p[:, 1]) * ((p[:, 2])))) / p.shape[0]
            x_hsi_loss = torch.sum(hsi_loss_batch * ((1 - p[:, 0]) * (p[:, 1]) * ((p[:, 2])))) / p.shape[0]
            x_auxi = x_dsm_loss + x_lidar_loss + x_hsi_loss

            if args.method == 'mse':
                kd_feature_loss = torch.mean(kd_feature_loss, dim=1)
            else:
                kd_feature_loss = torch.sum(kd_feature_loss, dim=1)
            # print(kd_feature_loss.shape)

            # print(kd_feature_loss.shape)

            teacher_whole_out_prob = F.softmax(teacher_out, dim=1)
            H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

            H_teacher_prob = H_teacher / torch.sum(H_teacher)
            kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)

            if epoch > 5:
                loss = cls_loss + kd_feature_loss * args.lambda_kd_feature + x_auxi * 0.5
            else:
                loss = cls_loss

            cls_loss_sum += cls_loss.item()
            kl_loss_sum += kl_loss_num.item()
            x_lidar_loss_sum += x_lidar_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        # testing
        student_model.p = [0, 0, 0]
        result_test = calc_accuracy_multi_tri(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_feature_loss_sum / (len(train_loader)),
              kl_loss_sum / (len(train_loader)), x_lidar_loss_sum / (len(train_loader)), )

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kl_loss_sum = 0
        x_lidar_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_knowledge_distill_fc_feature_center(net_dict, cost_dict, optimizer_dict, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    from loss.center_loss import CenterLoss
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']
    center_loss = cost_dict['center_loss']

    optimizer = optimizer_dict['optimizer_cls']
    optimizer_center = optimizer_dict['optimizer_ceter']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if args.method == 'mse':
        kd_loss_func = nn.MSELoss(reduction='none').cuda()
    elif args.method == 'dad':
        kd_loss_func = DAD().cuda()
    else:
        kd_loss_func = SP(reduction='none').cuda()

    mse_loss = nn.MSELoss().cuda()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    center_loss_sum = 0

    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            optimizer_center.zero_grad()

            student_out, x_origin, x_rec, student_feature = student_model(img_hsi, img_lidar)
            teacher_out, x2, x3, teacher_feature = teacher_model(img_hsi, img_lidar)

            cls_loss = criterionCls(student_out, target)

            rec_loss_1 = mse_loss(x_origin, student_out)
            rec_loss_2 = mse_loss(x_rec, student_out)
            rec_loss = rec_loss_1 + rec_loss_2
            cls_loss = cls_loss + rec_loss

            student_feature = student_feature.view(student_feature.shape[0], -1)
            teacher_feature = teacher_feature.view(student_feature.shape[0], -1)
            kd_feature_loss = kd_loss_func(student_feature, teacher_feature)

            center_loss_num = center_loss(student_feature, target)

            if args.method == 'mse':
                kd_feature_loss = torch.mean(kd_feature_loss, dim=1)
            else:
                kd_feature_loss = torch.sum(kd_feature_loss, dim=1)
            # print(kd_feature_loss.shape)

            # print(kd_feature_loss.shape)

            teacher_whole_out_prob = F.softmax(teacher_out, dim=1)
            H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)

            H_teacher_prob = H_teacher / torch.sum(H_teacher)
            kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)

            if epoch > 5:
                loss = cls_loss + kd_feature_loss * args.lambda_kd_feature
                loss = loss + center_loss_num * args.lambda_center_loss
            else:
                loss = cls_loss

                loss = loss + center_loss_num * args.lambda_center_loss / 10

            cls_loss_sum += cls_loss.item()
            center_loss_sum += center_loss_num.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer_center.step()

        # testing
        student_model.p = [0, 0]
        result_test = calc_accuracy_multi(model=student_model, args=args, loader=test_loader, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_feature_loss_sum / (len(train_loader)),
              center_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))

        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        center_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_knowledge_distill_dad(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP
    from loss.kd.sp import DAD, DAD_MA
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        pkt_loss = PKTCosSim().cuda()
    else:
        pkt_loss = PKTCosSim()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        dad_ma_loss = DAD_MA().cuda()
    else:
        dad_ma_loss = DAD_MA()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(100),
                                                                              np.int(250),
                                                                              np.int(350)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_m1 = img_m1.cuda()
                img_m2 = img_m2.cuda()
                target = target.cuda()

            # print(target)

            optimizer.zero_grad()

            x_whole_teacher, feature_teacher = teacher_model(img_m1, img_m2)

            if args.student_data == args.pair_modalities[0]:
                x_whole_student, feature_student = student_model(img_m1)
            else:
                x_whole_student, feature_student = student_model(img_m2)

            # print("time_forward:", time_forward.total_seconds())

            # logits蒸馏损失
            # if args.kd_mode in ['logits', 'st']:
            # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            # patch_loss = patch_loss.cuda()
            # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
            # whole_loss = whole_loss.cuda()
            # kd_loss = patch_loss + whole_loss
            kd_logits_loss = criterionKD(x_whole_student, x_whole_teacher.detach())

            # kd_logits_loss = bce_loss(student_patch_out, teacher_patch_out.detach())
            # kd_logits_loss = kd_logits_loss.cuda()
            # else:
            #     kd_logits_loss = 0
            #     print("kd_Loss error")

            # feature 蒸馏损失
            # student_layer3 = torch.mean(student_layer3, dim=1)
            # teacher_layer3 = torch.mean(teacher_layer3, dim=1)
            # kd_feature_loss = mse_loss(student_layer3, teacher_layer3)
            # student_layer3=torch.unsqueeze(student_layer3,dim=1)
            # kd_feature_loss_layer3 = sp_loss(student_layer3, teacher_layer3)
            # kd_feature_loss_layer4 = sp_loss(student_layer4, teacher_layer4)
            kd_feature_loss = dad_loss(feature_student, feature_teacher)

            # 分类损失
            cls_loss = criterionCls(x_whole_student, target)
            cls_loss = cls_loss.cuda()
            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature + kd_logits_loss * args.lambda_kd_logits

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=False)
        print(result_test)
        accuracy_test = result_test[0]

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader),
              kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_knowledge_distill_fkd(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP
    from loss.kd.sp import DAD
    from loss.measure import PA_Measure
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        pkt_loss = PKTCosSim().cuda()
    else:
        pkt_loss = PKTCosSim()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        pa_loss = PA_Measure().cuda()
    else:
        pa_loss = PA_Measure()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(100),
                                                                              np.int(250),
                                                                              np.int(350)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    dad_feature_loss_sum = 0
    mmd_loss_num = torch.tensor(0)
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    add_factor = torch.tensor(0)
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_m1 = img_m1.cuda()
                img_m2 = img_m2.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            x_whole_teacher, feature_teacher = teacher_model(img_m1, img_m2)

            if args.student_data == args.pair_modalities[0]:
                x_whole_student, feature_student = student_model(img_m1)
            else:
                x_whole_student, feature_student = student_model(img_m2)

            kd_feature_loss_2 = dad_loss(feature_student, feature_teacher)

            # print(feature_student.shape,feature_teacher.shape)
            feature_student = feature_student.view(feature_student.shape[0], -1)
            feature_teacher = feature_teacher.view(feature_teacher.shape[0], -1)
            kd_feature_loss_1 = pa_loss(feature_student, feature_teacher)
            if epoch > 10:
                mmd_loss_num = mmd_loss(feature_student, feature_teacher)
                #     add_factor = ((torch.sigmoid(mmd_loss_num) * 1) - 0.5) * 2
                add_factor = torch.exp(((mmd_loss_num - 0.6) / (0.6 ** 2)))
                if add_factor >= 1:
                    add_factor = torch.tensor(1.0)
                else:
                    add_factor = torch.tensor(0)
            add_factor = torch.tensor(0)
            kd_feature_loss = kd_feature_loss_1 * (1 - add_factor) + kd_feature_loss_2 * 15 * add_factor

            kd_logits_loss = criterionKD(x_whole_student, x_whole_teacher.detach())

            # 分类损失
            cls_loss = criterionCls(x_whole_student, target)
            cls_loss = cls_loss.cuda()
            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature + kd_logits_loss

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss_1.item()
            dad_feature_loss_sum += kd_feature_loss_2.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=False)
        print(result_test)
        accuracy_test = result_test[0]

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader),
              kd_feature_loss_sum / (len(train_loader)), dad_feature_loss_sum / (len(train_loader)),
              kd_logits_loss_sum / (len(train_loader)), add_factor.cpu().detach().numpy(),
              mmd_loss_num.cpu().detach().numpy())

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0
        dad_feature_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_knowledge_distill_jda(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    def PCA_svd(X, k, center=True):
        n = X.size()[0]
        ones = torch.ones(n).view([n, 1])
        h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
        H = torch.eye(n) - h
        H = H.cuda()
        X_center = torch.mm(H.double(), X.double())
        u, s, v = torch.svd(X_center)
        components = v[:k].t()
        components = components.float()
        # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
        return components

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP
    from loss.kd.sp import DAD, DAD_MA
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        pkt_loss = PKTCosSim().cuda()
    else:
        pkt_loss = PKTCosSim()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        dad_ma_loss = DAD_MA().cuda()
    else:
        dad_ma_loss = DAD_MA()

    dropout_process = nn.Dropout(0.5)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(100),
                                                                              np.int(250),
                                                                              np.int(350)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    dad_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_m1 = img_m1.cuda()
                img_m2 = img_m2.cuda()
                target = target.cuda()

            # print(target)

            optimizer.zero_grad()

            x_whole_teacher, feature_teacher = teacher_model(img_m1, img_m2)

            if args.student_data == args.pair_modalities[0]:
                x_whole_student, feature_student = student_model(img_m1)
            else:
                x_whole_student, feature_student = student_model(img_m2)

            # print("time_forward:", time_forward.total_seconds())

            # logits蒸馏损失
            # if args.kd_mode in ['logits', 'st']:
            # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            # patch_loss = patch_loss.cuda()
            # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
            # whole_loss = whole_loss.cuda()
            # kd_loss = patch_loss + whole_loss
            kd_logits_loss = criterionKD(x_whole_student, x_whole_teacher.detach())

            # kd_logits_loss = bce_loss(student_patch_out, teacher_patch_out.detach())
            # kd_logits_loss = kd_logits_loss.cuda()
            # else:
            #     kd_logits_loss = 0
            #     print("kd_Loss error")

            # feature 蒸馏损失
            # student_layer3 = torch.mean(student_layer3, dim=1)
            # teacher_layer3 = torch.mean(teacher_layer3, dim=1)
            # kd_feature_loss = mse_loss(student_layer3, teacher_layer3)
            # student_layer3=torch.unsqueeze(student_layer3,dim=1)
            # kd_feature_loss_layer3 = sp_loss(student_layer3, teacher_layer3)
            # kd_feature_loss_layer4 = sp_loss(student_layer4, teacher_layer4)
            dad_feature_loss = dad_loss(feature_student, feature_teacher)

            feature_teacher = feature_teacher.view(feature_teacher.shape[0], -1)
            feature_teacher_min = torch.min(feature_teacher, dim=1)[0]
            feature_teacher_min = torch.unsqueeze(feature_teacher_min, 1)
            feature_teacher_max = torch.max(feature_teacher, dim=1)[0]
            feature_teacher_max = torch.unsqueeze(feature_teacher_max, 1)
            # feature_teacher = (feature_teacher - feature_teacher_min) / (feature_teacher_max - feature_teacher_min)

            feature_student = feature_student.view(feature_student.shape[0], -1)
            feature_student_min = torch.min(feature_student, dim=1)[0]
            feature_student_min = torch.unsqueeze(feature_student_min, 1)
            feature_student_max = torch.max(feature_student, dim=1)[0]
            feature_student_max = torch.unsqueeze(feature_student_max, 1)
            # feature_student = (feature_student - feature_student_min) / (feature_student_max - feature_student_min)

            kd_feature_loss = mmd_loss(feature_student, feature_teacher)

            # 分类损失
            # x_whole_student=dropout_process(x_whole_student)
            cls_loss = criterionCls(x_whole_student, target)
            cls_loss = cls_loss.cuda()
            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature + kd_logits_loss * args.lambda_kd_logits + dad_feature_loss * 0

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()
            dad_feature_loss_sum += dad_feature_loss.item()

            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=False)
        print(result_test)
        accuracy_test = result_test[0]

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader),
              kd_feature_loss_sum / len(train_loader), dad_feature_loss_sum / len(train_loader),
              kd_logits_loss_sum / len(train_loader))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0
        dad_feature_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)
