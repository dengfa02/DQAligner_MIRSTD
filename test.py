import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from numpy import *
import os.path as osp
from tqdm import tqdm
from PIL import Image
import time
from MIRSDTDataLoader import TrainSetLoader, TestSetLoader
from IRDSTDataLoader import IRDST_TrainSetLoader, IRDST_TestSetLoader
from utils.metric_basic import *
from utils.loss import *
from model.DQAligner import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.autograd.set_detect_anomaly(True)


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Infrared_target_detection_overall')
    parser.add_argument('--DataPath', type=str, default='./dataset/', help='Dataset path [default: ./dataset/]')
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT',
                        help='Dataset name [dafult: NUDT-MIRSDT],IRDST,TSIRMT')
    parser.add_argument('--saveDir', type=str, default='./results/', help='Save path [defaule: ./results/]')
    parser.add_argument('--weight_path', type=str,
                        default='results/NUDT-MIRSDT/DQAligner/weight_NUDT-MIRSDT.pth',
                        help='model weight path')

    # train
    parser.add_argument('--model', type=str,
                        default='DQAligner_test_visual',
                        # 网络结构改了以后改这里名称即可
                        help='ResUNet_DTUM, DNANet_DTUM, ACM, ALCNet, ResUNet, DNANet, ISNet, UIU')
    parser.add_argument('--fullySupervised', default=True)
    parser.add_argument('--SpatialDeepSup', default=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_img', type=bool, default=False)  # 保存图像
    # loss & optimize
    parser.add_argument('--loss_func', type=str, default='adafocal',
                        help='HPM, FocalLoss, OHEM, fullySup, softiou, focal, focaliou, adafocal')
    parser.add_argument("--seed", type=int, default=42, help="seed")
    # GPU
    parser.add_argument('--DataParallel', default=False, help='Use one gpu or more')
    parser.add_argument('--mode', default='test', help='train or test')

    args = parser.parse_args()
    print("model: %s, dataset: %s" % (args.model, args.dataset))

    # the parser
    return args


def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


global args
args = parse_args()
seed_pytorch(args.seed)


def worker_init_fn(seed=42):
    # 确保每个worker都有不同的种子，但是基于主种子生成。
    # 这里使用worker_id作为扰动值以保证不同worker之间的随机性是独立的。
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_savepath(args, epoch, epoch_loss, CurTime):
    SavePath = args.saveDir + f'{args.dataset}/' + args.model + '_DeepSup' + str(
        args.SpatialDeepSup) + '_' + args.loss_func + f'_{CurTime}/'
    ModelPath = SavePath + 'Epoch_' + str(epoch) + '_' + str(
        epoch_loss) + '_loss_' + '.pth'  # 保存每轮模型
    ParameterPath = SavePath + f'weight_{args.model}' + '.pth'  # 保存最终

    if not os.path.exists(args.saveDir):
        os.makedirs(args.saveDir)
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)
    # if not os.path.exists(SavePath + f'{args.dataset}/'):
    #     os.makedirs(SavePath + f'{args.dataset}/')

    return ModelPath, ParameterPath, SavePath


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")  # args.device

        # model
        self.net = DQAligner(input_channels=1, num_frames=5, train_mode=True,
                             key_mode='last')  # Baseline_2D  , key_mode='mid'
        if args.DataParallel:
            self.net = nn.DataParallel(self.net, device_ids=[0, 1]).cuda()  # , device_ids=[0,1,2]).cuda()
        self.net = self.net.to(self.device)

        train_path = args.DataPath + args.dataset + '/'
        self.test_path = train_path
        if args.dataset == 'NUDT-MIRSDT':
            self.val_dataset = TestSetLoader(self.test_path, fullSupervision=args.fullySupervised)
        elif args.dataset in ['TSIRMT', 'IRDST']:
            self.val_dataset = IRDST_TestSetLoader(self.test_path, num_frames=5, fullSupervision=args.fullySupervised,
                                                   key_mode='last')
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0,
                                     pin_memory=False)

        self.PD_FA = PD_FA()
        self.mIoU = mIoU()
        self.ROC = ROCMetric(1, 20)
        self.F_metric = F_metric(nclass=1)
        self.PD_FA_cur = PD_FA()
        self.mIoU_cur = mIoU()
        self.ROC_cur = ROCMetric(1, 20)
        self.F_metric_cur = F_metric(nclass=1)

        self.best_fmeasure = 0
        self.best_iou = 0
        self.best_fa = 1e3

        self.loss_list = []
        self.epoch_loss = 0

        ########### save ############
        timestamp = time.time()
        self.StartTime = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime(timestamp))
        self.ModelPath, self.ParameterPath, self.SavePath = generate_savepath(args, 0, 0, self.StartTime)
        self.test_save = self.SavePath + 'visual_results/'
        # self.writeflag = 0  # 测试时也会写入metric，改为0，训练1
        self.save_img = args.save_img
        if self.save_img and not os.path.exists(self.test_save):
            os.makedirs(self.test_save)

    def validation(self, epoch):
        self.mIoU.reset()
        self.PD_FA.reset()
        self.F_metric.reset()
        self.fps_list = []
        self.ROC.reset()
        args = self.args
        if args.dataset == 'NUDT-MIRSDT':
            txt = np.loadtxt(self.test_path + 'test.txt', dtype=bytes).astype(str)
        else:
            txt = np.loadtxt(self.test_path + 'ImageSets/' + 'val_new.txt', dtype=bytes).astype(str)
        self.net.eval()

        cat_flag = 0
        feat_prop = None
        tbar = tqdm(self.val_loader)
        for i, data in enumerate(tbar, 0):
            with torch.no_grad():
                if args.dataset == 'NUDT-MIRSDT':
                    SeqData_t, TgtData_t, m, n = data
                    video_name = [txt[i].split('/')[0]]
                    frame_id = [txt[i].split('/')[-1]]
                else:
                    SeqData_t, TgtData_t, m, n, video_name, frame_id = data
                SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)

                time_start = time.time()
                outputs = self.net(SeqData, feat_prop, cat_flag, False)
                time_end = time.time()
                self.fps_list.append(1 / (time_end - time_start))

                if isinstance(outputs, (list, tuple)):
                    pred = outputs[1].squeeze(2)
                    feat_prop = outputs[2]
                else:
                    pred = outputs
                cat_flag = 1

                Outputs_Max = torch.sigmoid(pred)
                TestOut = Outputs_Max.data.cpu().numpy()[0, 0, 0:m, 0:n]

                if self.save_img and args.mode == 'test':
                    if args.dataset == 'NUDT-MIRSDT':
                        img = Image.fromarray(uint8((TestOut > 0.5) * 255))
                        folder_name = "%s%s/" % (self.test_save, txt[i].split('/')[0])
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        name = folder_name + txt[i].split('/')[-1].split('.')[0] + '.png'
                        img.save(name)

                    else:
                        img = Image.fromarray(uint8((TestOut > 0.5) * 255))
                        folder_name = "%s%s/" % (self.test_save, video_name[0])
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        name = folder_name + f'/{str(frame_id[0])}' + '.png'
                        img.save(name)

                """更新整个数据集指标"""
                self.mIoU.update((Outputs_Max[:, :, 0:m, 0:n].cpu() > 0.5),
                                 TgtData[:, :, :m, :n].cpu())  # 输入均为1,1,h,w
                self.PD_FA.update((Outputs_Max[0, 0, 0:m, 0:n].cpu() > 0.5), TgtData[0, 0, :m, :n].cpu(),  # 输入为h,w
                                  (m, n))
                self.F_metric.update((Outputs_Max[:, :, 0:m, 0:n].cpu() > 0.5), TgtData[:, :, :m, :n].cpu())
                self.ROC.update(Outputs_Max[:, :, 0:m, 0:n].cpu(), TgtData[:, :, :m, :n].cpu())
                results = self.mIoU.get()
                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, results[1]))

        print('FPS=%.3f', np.mean(self.fps_list))

        results1 = self.mIoU.get()
        results2 = self.PD_FA.get()
        prec, recall, fmeasure = self.F_metric.get()
        # ture_positive_rate, false_positive_rate = self.ROC.get()
        # roc_array = np.vstack((false_positive_rate, ture_positive_rate))
        # roc_save_dir = 'results/ROC/%s/SNR3-DQAligner-ROC.npy' % (args.dataset)
        # np.save(roc_save_dir, roc_array)

        """评估总体指标"""
        if args.mode == 'train':
            with open(osp.join(self.SavePath, 'metric_train.log'), 'a') as f:
                f.write("-" * 50)
                f.write(f"\nTotal results: \n")
                f.write('{} - {:03d}\t - IoU {:.5f}\t - F1 {:.5f}\t - PD {:.5f}\t - FA {:.5f}\n'.
                        format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                               epoch, results1[1], fmeasure, results2[0], results2[1] * 1000000))
                f.write("-" * 50)
            if fmeasure > self.best_fmeasure:
                self.best_fmeasure = fmeasure
                torch.save(self.net, self.SavePath + 'Epoch_%s_%.5f_best.pth' % (
                    epoch, self.best_fmeasure))  # 训练时最优模型保存

        if args.mode == 'test':
            print("pixAcc, mIoU:\t" + str(results1))
            print("PD, FA:\t" + str(results2))
            print('F_measure:\t' + str(fmeasure))

    def savemodel(self, epoch):
        self.ModelPath, self.ParameterPath, self.SavePath = generate_savepath(self.args, epoch, self.epoch_loss,
                                                                              self.StartTime)
        torch.save(self.net, self.ModelPath)
        print('save net OK in %s' % self.ModelPath)

    def saveloss(self):
        CurTime = time.strftime("%Y_%m_%d__%H_%M", time.localtime())
        print(CurTime)
        x1 = range(self.args.epochs)
        y1 = self.loss_list
        fig = plt.figure()
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        LossPNGSavePath = self.SavePath + 'train_loss_' + CurTime + '.png'
        plt.savefig(LossPNGSavePath)
        # plt.show()
        print('finished Show!')


if __name__ == '__main__':
    def compare_weights(model_a, model_b):
        diff_count = 0
        for (name_a, param_a), (name_b, param_b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
            if not torch.allclose(param_a, param_b, atol=1e-6):
                print(f"Weight different: {name_a}")
                diff_count += 1
        print(f"Total different params: {diff_count}")


    trainer = Trainer(args)
    if args.mode == 'test':
        #####################################################
        trainer.ModelPath = args.weight_path
        trainer.test_save = trainer.SavePath + 'visual_results/'
        # compare_weights(trainer.net, torch.load(trainer.ModelPath, map_location=trainer.device))
        trainer.net = torch.load(trainer.ModelPath, map_location=trainer.device)

        print('load weight OK!')
        epoch = args.epochs
        #####################################################
        trainer.validation(epoch)
