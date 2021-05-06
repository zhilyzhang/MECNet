import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.autograd import Variable
import time
import glob
from data_load import GeneratorData
from models.MECNet import *
from modules.loss import SSIMLoss, IOULoss
from modules.levelSetLoss import LSLoss
import warnings
import datetime
import logging
warnings.filterwarnings('ignore')

saveLog = 'logs'
os.makedirs(saveLog, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
fileName = datetime.datetime.now().strftime('day'+'%Y-%m-%d-%H')
handler = logging.FileHandler(os.path.join(saveLog, fileName + '.log'))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Train(object):
    def __init__(self, data_path, save_path, patch_size=512, epoch=100, batchSize=2):
        self.data_path = data_path
        self.save_path = save_path
        self.patch_size = patch_size
        self.epoch = epoch
        self.batchSize = batchSize

    def _train(self, epoch, model, batch_size=3):
        '''load data'''
        multi_scale = (1.0,)
        # multi_scale = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
        train_data_gen = GeneratorData(self.data_path + '/train_data/',
                                       batch_size=batch_size,
                                       multi_scale=multi_scale).generate(val=False)
        val_data_gen = GeneratorData(self.data_path + '/val_data/', batch_size=batch_size).generate(val=True)
        single_image_clip_num = 1  # 一张图裁剪为多少个瓦片
        single_image_clip_num_val = 1
        epoch_size_train = len(glob.glob(self.data_path + '/train_data/img/*.tif')) \
                           * len(multi_scale) * single_image_clip_num // batch_size

        epoch_size_val = len(glob.glob(self.data_path + '/val_data/img/*.tif')) * single_image_clip_num_val // batch_size
        num_flag = epoch_size_train // 100 if epoch_size_train // 100 == 0 else epoch_size_train // 100 + 1

        loss_func = torch.nn.BCEWithLogitsLoss()
        loss_func.cuda()
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        st_val_loss, st_train_loss = 10.0, 10.0
        st_val_acc = 0.
        start_time = time.time()

        for ep in range(1, epoch + 1):
            logger.info('doing epoch：{}'.format(ep))
            perLoss, perAcc = 0., 0.
            model.train()
            for idx in range(epoch_size_train):
                img, label = next(train_data_gen)
                # logger.info(img.shape)
                # logger.info(label.shape)
                img = Variable(img)
                img = img.cuda()
                output = model(img)

                label = Variable(label)
                label = label.cuda()

                loss = loss_func(output.squeeze(1), label)
                perLoss += loss.data.cpu().numpy()
                output = F.sigmoid(output)  # [0]
                predict = output.squeeze(1)

                predict[predict >= 0.5] = 1
                predict[predict < 0.5] = 0
                acc = (predict == label).sum().data.cpu().numpy() / (self.patch_size * self.patch_size * len(label))
                perAcc += acc
                optimizer.zero_grad()  # 清空上一步残留的更新参数值
                loss.backward()  # 误差反向传播，计算参数更新
                optimizer.step()  # 将参数更新值置于net的parameters中
                if idx % num_flag == 0:
                    logger.info('Train epoch: {} [{}/{} ({:.2f}%)]\tLoss:{:.6f}\tAcc:{:.6f}'.format(
                        ep, idx + 1, epoch_size_train, 100.0 * (idx + 1) / epoch_size_train,
                        loss.data.cpu().numpy(), acc))
            t_los_mean = perLoss / epoch_size_train
            t_acc_mean = perAcc / epoch_size_train
            logger.info('Train Epoch: {}, Loss: {:.4f}, Acc: {:.4f}'.format(ep, t_los_mean, t_acc_mean))
            ''' val '''
            model.eval()
            perValLoss = 0.
            perValAcc = 0.
            logger.info('正在进行验证模型，请稍等...')

            for idx in range(epoch_size_val):
                img, label = next(val_data_gen)
                with torch.no_grad():
                    img = Variable(img)
                    img = img.cuda()
                    output = model(img)
                    label = Variable(label)
                    label = label.cuda()
                    loss = loss_func(output.squeeze(1), label)

                perValLoss += loss.data.cpu().numpy()
                output = F.sigmoid(output)
                predict = output.squeeze(1)
                predict[predict >= 0.5] = 1
                predict[predict < 0.5] = 0
                valacc = (predict == label).sum().data.cpu().numpy() / (self.patch_size * self.patch_size * len(label))
                perValAcc += valacc
            val_los_mean = perValLoss / epoch_size_val
            val_acc_mean = perValAcc / epoch_size_val
            logger.info('Test Loss: {:.6f}, Acc: {:.6f}'.format(val_los_mean, val_acc_mean))

            if st_train_loss > t_los_mean and st_val_acc < val_acc_mean + 0.02 or st_val_acc < val_acc_mean:
                if st_train_loss > t_los_mean: st_train_loss = t_los_mean
                if st_val_acc < val_acc_mean: st_val_acc = val_acc_mean
                # 仅保存和加载模型参数
                logger.info('进行权重保存-->>\nEpoch：{}\t\nTrainLoss:{:.4f}\t\nValAcc:{:.4f}'
                      ''.format(ep, float(t_los_mean), float(val_acc_mean)))
                save_model = self.save_path + 'Epoch_{}_TrainLoss_{:.4f}_miou_{:.4f}.pkl'.format(
                    ep, float(t_los_mean), float(val_acc_mean))
                torch.save(model.state_dict(), save_model)
            duration1 = time.time() - start_time
            start_time = time.time()
            logger.info('train running time: %.2f(minutes)' % (duration1 / 60))

    def _train2(self, epoch, model, batch_size=3):
        '''load data'''
        multi_scale = (1.0,)
        # multi_scale = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
        train_data_gen = GeneratorData(self.data_path + '/train_data/',
                                       batch_size=batch_size,
                                       multi_scale=multi_scale).generate(val=False)
        val_data_gen = GeneratorData(self.data_path + '/val_data/', batch_size=batch_size).generate(val=True)

        single_image_clip_num = 1  # 一张图裁剪为多少个瓦片
        single_image_clip_num_val = 1
        epoch_size_train = len(glob.glob(self.data_path + '/train_data/img/*.tif')) \
                           * len(multi_scale) * single_image_clip_num // batch_size

        epoch_size_val = len(glob.glob(self.data_path + '/val_data/img/*.tif')) * single_image_clip_num_val // batch_size
        num_flag = epoch_size_train // 100 if epoch_size_train // 100 == 0 else epoch_size_train // 100 + 1

        loss_func = torch.nn.BCEWithLogitsLoss()
        loss_func.cuda()
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        st_val_acc = 0.
        start_time = time.time()

        for ep in range(1, epoch + 1):
            logger.info('doing epoch：{}'.format(ep))
            perLoss, perAcc = 0., 0.
            model.train()
            for idx in range(epoch_size_train):
                img, label = next(train_data_gen)
                img = Variable(img)
                img = img.cuda()
                output = model(img)
                label = Variable(label)
                label = label.cuda()
                loss = loss_func(output[0].squeeze(1), label)
                loss1 = loss_func(output[1].squeeze(1), label)
                loss2 = loss_func(output[2].squeeze(1), label)
                loss3 = loss_func(output[3].squeeze(1), label)
                loss4 = loss_func(output[4].squeeze(1), label)
                loss5 = loss_func(output[5].squeeze(1), label)
                loss = 0.5 * loss + (loss1 + loss2 + loss3 + loss4 + loss5) * 0.5
                perLoss += loss.data.cpu().numpy()
                output = F.sigmoid(output[0])
                predict = output.squeeze(1)
                # logger.info(predict.shape, label.shape)
                predict[predict >= 0.5] = 1
                predict[predict < 0.5] = 0
                acc = (predict == label).sum().data.cpu().numpy() / (self.patch_size * self.patch_size * len(label))
                perAcc += acc
                optimizer.zero_grad()  # 清空上一步残留的更新参数值
                loss.backward()  # 误差反向传播，计算参数更新
                optimizer.step()  # 将参数更新值置于net的parameters中
                if idx % num_flag == 0:
                    logger.info('Train epoch: {} [{}/{} ({:.2f}%)]\tLoss:{:.6f}\tAcc:{:.6f}'.format(
                        ep, idx + 1, epoch_size_train, 100.0 * (idx + 1) / epoch_size_train,
                        loss.data.cpu().numpy(), acc))
            t_los_mean = perLoss / epoch_size_train
            t_acc_mean = perAcc / epoch_size_train
            logger.info('Train Epoch: {}, Loss: {:.4f}, Acc: {:.4f}'.format(ep, t_los_mean, t_acc_mean))
            ''' val '''
            model.eval()
            perValLoss = 0.
            perValAcc = 0.
            logger.info('正在进行验证模型，请稍等...')
            for idx in range(epoch_size_val):
                img, label = next(val_data_gen)
                with torch.no_grad():
                    img = Variable(img)
                    img = img.cuda()
                    output = model(img)
                    label = Variable(label)
                    label = label.cuda()
                    loss = loss_func(output[0].squeeze(1), label)
                    loss1 = loss_func(output[1].squeeze(1), label)
                    loss2 = loss_func(output[2].squeeze(1), label)
                    loss3 = loss_func(output[3].squeeze(1), label)
                    loss4 = loss_func(output[4].squeeze(1), label)
                    loss5 = loss_func(output[5].squeeze(1), label)
                    loss = 0.5 * loss + (loss1 + loss2 + loss3 + loss4 + loss5) * 0.5

                perValLoss += loss.data.cpu().numpy()
                output = F.sigmoid(output[0])
                predict = output.squeeze(1)
                predict[predict >= 0.5] = 1
                predict[predict < 0.5] = 0
                valacc = (predict == label).sum().data.cpu().numpy() / (self.patch_size * self.patch_size * len(label))
                perValAcc += valacc

            val_los_mean = perValLoss / epoch_size_val
            val_acc_mean = perValAcc / epoch_size_val

            logger.info('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_los_mean, val_acc_mean))

            if st_val_acc < val_acc_mean:
                st_val_acc = val_acc_mean
                # 仅保存和加载模型参数
                logger.info('进行权重保存-->>\nEpoch：{}\t\nTrainLoss:{:.4f}\t\nValAcc:{:.4f}'
                      ''.format(ep, float(t_los_mean), float(val_acc_mean)))
                save_model = self.save_path + 'best_weights.pkl'
                torch.save(model.state_dict(), save_model)
            duration1 = time.time() - start_time
            start_time = time.time()
            logger.info('train running time: %.2f(minutes)' % (duration1 / 60))

    def train(self):
        model = MECNet()
        self._train2(self.epoch, model=model, batch_size=self.batchSize)


if __name__ == '__main__':
    data_path = r'/home/zzl/datasets/water-body-dataset'
    save_path_name = 'res/MECNet/'
    save_path = data_path + '/' + save_path_name
    os.makedirs(save_path, exist_ok=True)
    TN = Train(data_path, save_path, epoch=100,  batchSize=4)
    TN.train()
