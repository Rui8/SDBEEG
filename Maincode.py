import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset
import warnings
import os
from sklearn.model_selection import StratifiedKFold
import pickle
from utils.myaml import load_config
from torch.nn.utils import weight_norm
warnings.filterwarnings('ignore')
import optuna
import scipy.io
from model import DCN
from train import objective1

def valid(net, X_val, y_val):
    matrix='ABCDEF'+'GHIJKL'+'MNOPQR'+'STUVWX'+'YZ1234'+'56789_'
    targetTest=['N','E','U','R','A','L','_','N','E','T','W','O','R','K','S','_','A','N','D','_','D','E','E','P','_',
                'L','E','A','R','N','I','N','G']
    oder = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y',
            'Z','1','2','3','4','5','6','7','8','9','_']
    pwd = os.getcwd()
    fathpath=os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
    label = scipy.io.loadmat(fathpath+'/feature/random_cell_order.mat')
    rc_order = label['rc_order']
    input1 = X_val
    label = y_val
    rela = np.zeros((60,1))
    net.eval()
    batch_size = 60
    ss = int(input1.shape[0]/1980)
    t_char2 = np.zeros((ss,33,5),dtype=str)
    TRAIL=1980
    for su in range(ss):
        inputs = torch.tensor(input1[su*TRAIL:(su+1)*TRAIL,:,:,:])
        labels = torch.tensor(label[su*TRAIL:(su+1)*TRAIL,:])
        test_loader = DataLoader(TensorDataset(inputs.cuda(), labels.cuda()), batch_size=batch_size)
        accu = np.zeros((33,1))
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(test_loader, 0):#33个字符 循环
                nc = 0
                nSeq = 0
                in_nc = 0
                channel = 62
                dat = np.zeros((36,1,channel,80))
                tm_dat = np.zeros((36,1,channel,80))
                b = np.zeros((1,channel,80))

                for i in range(60):
                    loca = oder.index(targetTest[step])
                    for i2 in range(6):
                        si = dat.shape[1]
                        if dat[np.array(rc_order[0,nSeq][in_nc,i2])-1,si-1,0,0].all()==0:
                            aa = np.where(dat[np.array(rc_order[0,nSeq][in_nc,i2])-1,:,0,0]==0)
                            dat[np.array(rc_order[0,nSeq][in_nc,i2])-1,aa[0][0],:,:] = batch_x[nc,:,:,:].data.cpu()
                        else:
                            dat = np.insert(dat,si,values=b,axis=1)
                            dat[np.array(rc_order[0,nSeq][in_nc,i2])-1,si,:,:] = batch_x[nc,:,:,:].data.cpu()#36*1*32*80

                    for i3 in range(36):
                        if dat[i3,:,0,0].shape[0]==1:
                            tm_dat[i3,:,:,:] = dat[i3,:,:,:]
                        else:
                            bb = np.where(dat[i3,:,0,0]==0)
                            if np.array(bb).sum()==0:
                                tm_dat[i3,:,:,:] = np.expand_dims((np.mean(dat[i3,:,:,:],axis=0)),axis=0)
                            else:
                                tm_dat[i3,:,:,:] = np.expand_dims((np.mean(dat[i3,:bb[0][0],:,:],axis=0)),axis=0)
                    tm_dat1 = Variable(torch.from_numpy(tm_dat)).cuda()
                    tm_dat1 = tm_dat1.type(torch.cuda.FloatTensor)
                    y_pred = net(tm_dat1).cuda()
                    yy = y_pred[:,0].data.cpu()
                    b0 = np.where(yy==yy.min())
                    t_char2[su,step,nSeq] = matrix[b0[0][0]]
                    if loca+1 in np.array(rc_order[0,nSeq][in_nc,:]):
                        rela[i,0] = 1
                    else:
                        rela[i,0] = 0
                    nc+=1
                    in_nc+=1
                    if in_nc>11:
                        in_nc=0
                        nSeq+=1
                y_pred1 = net(batch_x).cuda()
                yy1 = y_pred1.data.cpu()
                b0 = torch.max(yy1,dim = 1)[1]#预测标签
                b1 = np.expand_dims(b0,axis=1)
                accu[step] = np.mean((rela == b1).astype(int))

    num_repeats = 5
    accTest = np.zeros((ss,num_repeats))
    for i in range(num_repeats):
        for sss in range(ss):
            accTest[sss,i] = np.mean(np.array(t_char2[sss,:,i] == targetTest).astype(int))
    acc = np.mean(accTest,axis=0)
    acc1 = np.mean(accu,axis = 0)

    return acc,acc1

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    sub = 2
    trial = 660*3
    fllag = False
    torch.set_num_threads(1)
    pwd = os.getcwd()
    fathpath = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")

    ## 导入全部的数据
    for s in range(sub):
        # print(s+1)
        with open(fathpath+'/feature/xxx'+str(s+1)+'.pickle', 'rb') as f:
            x = pickle.load(f)
        with open(fathpath+'/feature/yyy'+str(s+1)+'.pickle', 'rb') as f:
            y = pickle.load(f)
        if not fllag:
            x_data = x
            y_label = y.T
            fllag = True
        else:
            x_data = np.concatenate((x_data,x))
            y_label = np.concatenate((y_label,y.T))
            ## 开始交叉验证
    # print('Start cross validation')
    fv2 = torch.randn(sub,1980,1,62,80)
    y = np.zeros((sub,1980))
    del x_data, y_label

    skf = StratifiedKFold(n_splits=sub)#三折交叉验证

    skf.get_n_splits(fv2,y)
    ree = np.zeros((sub,6))

    running_loss_array = []
    result = []
    #迭代的次数一共为3*50=150
    step = 1
    #参数配置
    config = load_config('./config.yaml')
    n_trials = config.n_trials
    save_path = os.path.join(config.results_save_path, 'Deepmodel')#上层路径results
    for train_index,test_index in skf.split(fv2,np.ones((sub,1))):#所有被试的交叉验证 第一次
        X_train,X_val = fv2[train_index],fv2[test_index]
        y_train = y[:sub-1,:]
        y_val = y[sub-1,:]
        torch.cuda.synchronize()

        print('===========subject %d===========' % (step))
        #参数配置
        db_save_path = os.path.join(save_path, 'study')
        if not os.path.exists(db_save_path):
            os.makedirs(db_save_path)
        study_name = 'Deepmodel_subject'+str(step)
        study_name = os.path.join(db_save_path, study_name)
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name,
                                    direction='maximize',
                                    pruner=optuna.pruners.SuccessiveHalvingPruner())
        study.optimize(lambda trial:
            objective1(trial,X_train,y_train,step,sub,config), n_trials=n_trials)

        best_model = DCN('elu').cuda()
        model_save_path = os.path.join(config.results_save_path, 'Deepmodel', 'models')
        best_model_name = os.path.join(model_save_path, 'Deepmodel'+str(step)+".pth")
        best_model.load_state_dict(torch.load(best_model_name))
        #将最好的模型用在一个被试上进行测试
        acc_test1,acc_test2 = valid(best_model, X_val[0], np.expand_dims(y_val,axis=1))
        ree[step-1,:5] = acc_test1
        print(acc_test1)
        step+= 1



























