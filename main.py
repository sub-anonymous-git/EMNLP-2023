import numpy as np
import argparse, time, pickle
import torch
import numpy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from model import LSTMModel, MaskedNLLLoss
from dataloader import *
from fusion import fuse
from model import Transformer, MaskedNLLLoss
import config
from sklearn.model_selection import train_test_split
from Bert_Gcn import mf as mf

############################################################################### This code contain Both 80:20 and Cross Validation############################### 

def collate_batch(batch):
    text,acus,vid,label,umask = list(),list(),list(),list(),list()
    for t,a,v,l,mask in batch:
        text.append(t)
        acus.append(a)
        vid.append(v)
        label.append(l)
        umask.append(mask)
    
    return pad_sequence(text, padding_value=2.0), pad_sequence(acus, padding_value=2.0),\
    pad_sequence(vid, padding_value=2.0),pad_sequence(label,True),pad_sequence(umask, padding_value=2)

def get_our_loaders(path1,path2, batch_size=32,  num_workers=0, pin_memory=False):
    trainset = load_dataset(path1)
    #validset = load_dataset(path2)
    testset  = load_dataset(path2)
    print(len(trainset))
    lengths = [int(len(trainset)*0.8), int(len(trainset)*0.2)]
    trainset, validset = torch.utils.data.random_split(list(trainset), lengths)

   
    train_loader = DataLoader(list(trainset),
                              batch_size=batch_size,
                              collate_fn=collate_batch,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(list(validset),
                              batch_size=batch_size,
                              collate_fn=collate_batch,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(list(testset),
                             batch_size=batch_size,
                             collate_fn=collate_batch,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader,valid_loader,test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for i, data in enumerate(dataloader):
        
        if train:
            optimizer.zero_grad() 
        textf, image,  label,umask = data[0].cuda(),data[1].cuda(),data[2].cuda(),data[3].cuda()
        
        # print("i am label",label[1])
        # print("i am text",len(textf))
        
        #textf, acouf, qmask, umask, label =[d.cuda() for d in data[:-1]] if cuda else data[:-1]
        
       # **************************************************#
       
        #y=ZEE(100).cuda()
        #log_prob=y(textf, visuf,acouf,len(textf))
       
        #log_prob, alpha, alpha_f, alpha_b = model(log_prob, qmask, umask)
        #print("i am hidden",log_prob.shape)
        
        #print("i am done",log_prob.shape)

        #***********************************#
        
        
        #print("ia m shape",umask)
        #print("i am shape",torch.cat((textf, acouf, visuf), dim=-1).shape)
        # print("shape i am FUCK UUUU",torch.cat((textf, acouf, visuf), dim=-1).shape) 
        ### using simple concatination######## 
        image_sent=SparseAutoencoder(sent_image)
        log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, image,image_sent), dim=-1), umask)
        ###using mf###################

        log_prob, alpha, alpha_f, alpha_b = model(mf((textf, image,image_sent), dim=-1), umask)
        #log_prob, alpha, alpha_f, alpha_b = model(textf, umask)
        

         
        #***********************************************************************#

        #print("i am prediction",log_prob)

        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        # print(lp_)
        

        
        labels_ = label.view(-1)
        print("labels_",labels_.shape)
        print("lp_",lp_.shape)
        print("umask",umask.shape)
    
        
        loss = loss_function(lp_, labels_, umask)

        # pred_ = torch.argmax(lp_, 1) 
        #print("i am prediction",preds)

        preds.append(lp_.data.cpu().numpy())
        
        labels.append(labels_.data.cpu().numpy())
       
        masks.append(umask.reshape(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    preds = np.array(preds) >0.4
   
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]



for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = FNetEncoderLayer()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    for epoch in range(num_epochs):
        train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
        test_loss, test_correct=valid_epoch(model,device,test_loader,criterion)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                             num_epochs,
                                                                                                             train_loss,
                                                                                                             test_loss,
                                                                                                             train_acc,
                                                                                                             test_acc))
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    foldperf['fold{}'.format(fold+1)] = history  

torch.save(model,'k_cross.pt') 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.06, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=160, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weight')
    parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    batch_size = args.batch_size
    cuda       = args.cuda
    n_epochs   = args.epochs
    
    n_classes  = 2
    D_m = 2524
    D_e = 300
    D_h = 100

    # model = LSTMModel(D_m, D_e, D_h,
    #                   n_classes=n_classes,
    #                   dropout=args.dropout,
    #                   attention=args.attention)
    model = Transformer(D_m,D_e,D_h,attention=args.attention)
    if cuda:
        model.cuda()
        
    loss_weights = torch.FloatTensor([ 1.0,  1.0])
    
    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    train_loader,valid_loader,test_loader = get_our_loaders('train_output.pkl','test_output.pkl',batch_size=batch_size)
  

    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        # print("hi there")
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function,
                                               train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, val_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss == None or best_loss > test_loss:
             best_loss, best_label, best_pred, best_mask, best_attn =\
                     test_loss, test_label, test_pred, test_mask, attentions

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)
        print('epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                 format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                         test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
    if args.tensorboard:
        writer.close()

    print('Test performance..')
    from sklearn.metrics import f1_score
    l=f1_score(best_label, best_pred, average='micro')
    print("i am everything",l)
    print("best label",best_label)
    print("best pred",best_pred)
    #print('Loss {} F1-score {}'.format(best_loss,round(f1_score(best_label, best_pred, sample_weight=best_mask, average='weighted')*100, 2)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

  

